import sympy 
import numpy as np 

global_varnames = [] 
def SVGen(): # slack vaname generator 
    i = 0 
    while True:
        sym = sympy.Symbol('slack_'+str(i))
        global_varnames.append(sym) 
        yield sym 
        i += 1
global_svgen = SVGen() 


Ops = {'<':-2, '<=':-1, '=':0, '>=':1, '>':2} 
class Op: 
    def __init__(self, op): 
        if (isinstance(op, int)): 
            self.sym = list(Ops.keys())[op] 
            self.num = op 
        elif (isinstance(op, str)): 
            self.sym = op 
            self.num = Ops[op] 
        else: 
            raise ValueError("op must be int or str! Got "+str(op))
    
    def __eq__(self, op2): 
        if (isinstance(op2, Op)): 
            return op2.num == self.num 
        else: 
            return Op(op2).num == self.num

    def __str__(self):
        return "<Op '{}' ({})>".format(self.sym, self.num) 

class LinIneq: # linear only 
    def __init__(self, lhs, op, rhs): 
        self.lhs = lhs 
        self.op = op 
        self.rhs = rhs 

        self.simp_lhs = lhs - rhs # simplify to lhs (op) 0 

        self.eqz_gez_repr = None 
    
    def get_eqz_gez_repr(self, svgen): 
        # returns (expr, newvar) 
        # those newvars will be the ones >= 0 
        # lhs includes the constants though 
        if self.eqz_gez_repr is not None: 
            return self.eqz_gez_repr 
        
        if self.op == Op('='): 
            self.eqz_gez_repr = (self.simp_lhs.expand(), None) 
        elif self.op == Op('<='): 
            s_var = next(svgen) 
            self.eqz_gez_repr = ((self.simp_lhs+s_var).expand(), s_var)
        elif self.op == Op('>='): 
            s_var = next(svgen) 
            self.eqz_gez_repr = ((self.simp_lhs-s_var).expand(), s_var)
        else: 
            raise ValueError("Operations < and > are not allowed! Got {}".format(self.op)) 
        
        return self.eqz_gez_repr 
    
    @classmethod 
    def eqz_to_vec(cls, eqz, varnames, return_const = True): 
        coefs = eqz.as_coefficients_dict() 
        if return_const: 
            return [coefs[v] for v in varnames], coefs[1] 
        return [coefs[v] for v in varnames] 
    # did not negate the const, so it's all just one side only 
        


class Linear: 
    def __init__(self): 
        self.varnames = [] 
        self.varname_to_idx = {} 

        self.eqz = [] # linear expressions that are = 0 
        self.gez = [] # vars that are >= 0 

        # Ax = b 
        self.A = np.array([]) 
        self.b = np.array([]) 

    def add_var(self, varname): 
        self.varname_to_idx[varname] = len(self.varnames) 
        self.varnames.append(varname)

    def add_vars(self, varnames):
        for varname in varnames:
            self.add_var(varname) 
    
    def add_constraint(self, ineq:LinIneq, svgen=None): 
        eq, gez = ineq.get_eqz_gez_repr(svgen)
        if gez is not None: 
            self.add_var(gez) 
            self.gez.append(gez) 
        self.eqz.append(eq) 

        
    def eqz_to_Ab(self): 
        A = [] 
        b = [] 
        for eqz in self.eqz: 
            vec, const = LinIneq.eqz_to_vec(eqz, self.varnames, return_const=True) 
            A.append(vec) 
            b.append(-const) 

        self.A = np.array(A, dtype=np.float32) 
        self.b = np.array(b, dtype=np.float32) 
    
    def clean_Ab(self): 
        # make A's rows linearly independent 
        # then, since row rank is equal to column rank, there will be enough linearly independent columns. 
        lin_indps = [] 
        new_b = [] 

        _, inds = sympy.Matrix(self.A).T.rref() # get linearly independent rows 

        inds = list(inds) # so that it isn't a tuple / for each dim, but one dim only 

        # only keep those rows 
        self.A = self.A[inds] 
        self.b = self.b[inds]

    def get_bfs0(self): 
        # IMPLICIT CHOICE OF WHICH ROWS TO USE FOR BASIC FEASIBLE SOLUTION - FIRST ONES 
        _, col_inds = sympy.Matrix(self.A).rref() # get linearly independent columns
        col_inds = list(col_inds) 
        # col_inds will be the chosen linearly independent ones 

        # consider only the nonzero ones 
        Am = self.A[:, col_inds]
        x_B = np.linalg.inv(Am) @ self.b
        # convert back to x 
        x = np.zeros_like(self.A[0])
        x[col_inds] = x_B

        return x, col_inds # basic feasible solution found

    def x_to_subsdict(self, x):
        sd = {} 
        for vidx in range(len(x)):
            sd[self.varnames[vidx]] = x[vidx]
        return sd 

    # let there be n linearly independent rows, m columns. 

if __name__=='__main__':
    varname_syms = sympy.symbols("x1:6") 

    for v in varname_syms:
        global_varnames.append(v)
    
    cond1lhs = sympy.sympify("x1 + x2 + 2*x3 + x4")
    cond1rhs = sympy.sympify("1")
    cond1 = LinIneq(cond1lhs, Op('>='), cond1rhs)

    cond2lhs = sympy.sympify('3*x1 + x2 - x5')
    cond2rhs = sympy.sympify('3')
    cond2 = LinIneq(cond2lhs, Op('='), cond2rhs)

    cond3lhs = sympy.sympify("x2 + x4 + 5*x5")
    cond3rhs = sympy.sympify("-3 + x1")
    cond3 = LinIneq(cond3lhs, Op('<='), cond3rhs)

    cond4lhs = sympy.sympify('x1 + 2*x3 - 5*x5')
    cond4rhs = sympy.sympify('5')
    cond4 = LinIneq(cond4lhs, Op('>='), cond4rhs)

    cost = sympy.sympify('x1 + x2 + x3 + x4 + x5') # NOTE: if updating, update costvec too 

    lin = Linear()
    lin.add_vars(varname_syms) 
    lin.add_constraint(cond1, svgen=global_svgen)
    lin.add_constraint(cond2, svgen=global_svgen)
    lin.add_constraint(cond3, svgen=global_svgen)
    lin.add_constraint(cond4, svgen=global_svgen)

    for v in varname_syms:
        lin.gez.append(v) 

    # prep to solve 
    lin.eqz_to_Ab() 
    lin.clean_Ab()

    # make cost evaluator vector 
    costvec = np.array([1 if i<5 else 0 for i in range(len(lin.varnames))], dtype=np.float32) 

    # start solving - get first BFS 
    x0, inds0 = lin.get_bfs0()
    c0 = cost.subs(lin.x_to_subsdict(x0)) # we didn't actually use this oops

    print("FIRST BFS FOUND! VALUES:")
    print("X0", x0)
    print("INDS0", inds0)
    print("COST", np.dot(costvec, x0))
    print() 
    

    while True: # while haven't found solution yet 

        # calculate change in cost for each direction
        dcosts = {} # dcost: (in num, out num, new x)
        for dirn_no in range(len(lin.varnames)):
            if dirn_no in inds0: continue # don't choose already done ones
            # now, we've chosen to move in this direction - which means go let this =0 thing become not 0
            # we'll be going in the direction until smtg else becomes 0 (or if it goes to infinity then min cost is -inf)
            # if we move in this direction, it causes an increase in Ax, so we need to change the rest to compensate.
            # If we just move until x_dirn_no becomes 1, we get delta b:
            db = lin.A[:, dirn_no]
            # so, we need to find drest(m,1) which represents how much change in each in each column with index in inds0
            # such that it ends up with -db, to counteract this change.
            # basis(m,m) @ drest(m,1) = -db
            # drest(m, 1) = basis^-1(m,m) @ (-db) = -basis^-1(m,m) @ db
            # represent basis with B.
            B = lin.A[inds0]
            drest = - np.linalg.inv(B) @ db

            # find out which ones are first to reach end. x_i/-drest_i is smallest 
            dists = x0 / -drest # get smallest nonnegatives value 
            awayidx = np.argmin(np.where(dists>0, dists, np.Inf)) # hope that floating point errors can be ignored, so it will be min index and there's no looping
            # if there is looping, degenerate cases, then use lexicograpic or something TODO

            if dists[awayidx] == np.Inf:
                # that means that cost can be negative infinity
                print("COST CAN BE NEGATIVE INFINITY!!")
                1/0 # TODO 


            # get change in cost after moving - increase db component and drest component 
            # (dists[awayidx] * (db+)
            dx = np.zeros_like(self.A[0])
            dx[inds0] = dists[awayidx] * drest
            dx[dirn_no] = dists[awayidx] # * 1 
            dcost = np.dot( costvec , dx )

            # if cost reduces
            if dcost < 0:
                # convert awayidx to the normal x coordinates
                away_idx = inds0[awayidx] 
                dcosts[dcost] = (dirn_no, away_idx, dx) # save this direction 

        dcs = dcosts.keys() 
        if len(dcs) == 0:
            # optimal already, at a local (global) minimum
            print("OPTIMIZED YAY")
            1/0

        # get min
        mindc = min(dcs)
        inidx, outidx, dx = dcosts[mindc]

        # replace
        outidx_idx = inds0.index(outidx)
        inds0[outidx_idx] = inidx
        x0 += dx

        # should be done. Can wait for next loop 
        print("LOOP DONE! NEW VALUES:")
        print("X0", x0)
        print("INDS0", inds0)
        print("COST", np.dot(costvec, x0))
        print() 
