import sympy 
import numpy as np 
EPS = 1e-10 

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
        if self.num < -1 or self.num > 1: 
            raise ValueError("< or > OPS are currently not allowed")
    
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

    def __str__(self):
        return "<LinIneq: {} {} {}>".format(self.lhs, self.op.sym, self.rhs)  


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

    def get_bfs0(self, aux_print_every_new_bfs=True, aux_print_every_pivot_dirn=False, aux_print_normal=True):
        '''
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
        ''' # that was wrong.
        #return np.array([0,0,0,20,20,20], dtype=np.float32), [3,4,5] 

        # we shall create a different easy-to-solve problem. 
        A = np.concatenate([self.A, np.diag((2*((self.b>0) - 1/2)))], axis=1) # not np.eye but this because negatives if needed 
        x0 = np.concatenate([np.zeros(self.A.shape[1]), abs(self.b.copy())]) # this is a basic feasible solution 
        inds0 = [i for i in range(self.A.shape[1], A.shape[1])] 
        costvec = x0.copy() 

        if aux_print_normal: 
            print("FINDING BFS0") 

        lin = self # so that it can be copy-pasted 

        while True: # while haven't found solution yet

            # calculate change in cost for each direction
            final_dcost, final_inidx, final_outidx, final_dx = None, None, None, None 
            for dirn_no in range(len(lin.varnames)):
                if dirn_no in inds0: continue # don't choose already done ones
                # now, we've chosen to move in this direction - which means go let this =0 thing become not 0
                # we'll be going in the direction until smtg else becomes 0 (or if it goes to infinity then min cost is -inf)
                # if we move in this direction, it causes an increase in Ax, so we need to change the rest to compensate.
                # If we just move until x_dirn_no becomes 1, we get delta b:
                db = A[:, dirn_no]
                # so, we need to find drest(m,1) which represents how much change in each in each column with index in inds0
                # such that it ends up with -db, to counteract this change.
                # basis(m,m) @ drest(m,1) = -db
                # drest(m, 1) = basis^-1(m,m) @ (-db) = -basis^-1(m,m) @ db
                # represent basis with B.
                B = A[:,inds0]
                drest = - np.linalg.inv(B) @ db
                # we only need to consider drest < 0
                # NOTE TO SELF: if i decide to consider all again, remember the corner case
                # about x0[inds0][?]=0 but drest>0 so it's actually inf, but it's 0 in the dist calculation

                dists = np.where(drest<0, x0[inds0] / -drest, np.Inf)
                awayidx = np.argmin(dists) 
                

                if dists[awayidx] == np.Inf:
                    # that means that cost can be negative infinity
                    print("COST OF AUXILIARY CAN BE NEGATIVE INFINITY!!")
                    1/0 # TODO

                #print("DISTS:", dists) 


                # get change in cost after moving - increase db component and drest component 
                # (dists[awayidx] * (db+)
                dx = np.zeros_like(A[0])
                dx[inds0] = dists[awayidx] * drest
                dx[dirn_no] = dists[awayidx] # * 1 
                dcost = np.dot( costvec , dx )

                if aux_print_every_pivot_dirn: 
                    # debug 
                    print("PIVOT DIRN", dirn_no)
                    print("DX:", dx)
                    print("DCOST:", dcost)
                    print("DISTS:", dists)
                    print("DREST:", drest)
                    print() 

                # check to make sure the >=0 constraints will still be fulfilled 

                # if cost ~doesnt increase~ stricly decreases
                # what about basis changes? 
                if dcost < -EPS: #= EPS:
                    # convert awayidx to the normal x coordinates
                    away_idx = inds0[awayidx]
                    final_dcost = dcost
                    final_inidx = dirn_no
                    final_outidx = away_idx
                    final_dx = dx
                    break # found this, since it's lowest index, take it 
                
                    #dcosts[dcost] = (dirn_no, away_idx, dx) # save this direction 

            if final_dcost is None: 
                # optimal already, at a local (global) minimum
                if aux_print_normal: 
                    print("OPTIMIZED AUXILIARY")
                    print("X0", x0)
                    print("INDS0", inds0)

                c = np.dot(costvec, x0) 
                if abs(c) > EPS:
                    print("NOT A BFS! COST IS STILL", c) 
                    1/0

                # prep inds0: remove 0s that linger
                zeros = [] 
                for i in range(len(x0)):
                    if abs(x0[i]) < EPS:
                        zeros.append(i)
                alr_replaced_cnt = 0 
                for i in range(len(inds0)):
                    if abs(x0[inds0[i]]) < EPS:
                        inds0[i] = zeros[alr_replaced_cnt]
                        alr_replaced_cnt += 1
                # just verify
                if (2*alr_replaced_cnt > len(zeros)):
                    print("IT SEEMS THERE'S LESS ZEROS THAN REALLY?")
                    print("new X0:", x0)
                    print("new INDS0", inds0)
                    1/0 

                if aux_print_normal: 
                    print()
                    print() 
                # return the bfs relevant
                return x0[:self.A.shape[1]], inds0 

            # replace
            outidx_idx = inds0.index(final_outidx)
            inds0[outidx_idx] = final_inidx
            x0 += final_dx

            # should be done. Can wait for next loop 
            if aux_print_every_new_bfs:
                print("LOOP DONE! NEW VALUES:")
                print("X0", x0)
                print("INDS0", inds0)
                print("COST", np.dot(costvec, x0))
                #print(lin.b) 
                print()
            



    def x_to_subsdict(self, x):
        sd = {} 
        for vidx in range(len(x)):
            sd[self.varnames[vidx]] = x[vidx]
        return sd 

    # let there be n linearly independent rows, m columns. 

if __name__=='__main__':
    print_every_pivot_dirn = False
    print_every_new_bfs = True 
    varname_syms = sympy.symbols("x1:4") 

    for v in varname_syms:
        global_varnames.append(v)
    
    cond1lhs = sympy.sympify("x1 + 2*x2 + 2*x3")
    cond1rhs = sympy.sympify("20")
    cond1 = LinIneq(cond1lhs, Op('<='), cond1rhs)

    cond2lhs = sympy.sympify('2*x1 + x2 + 2*x3')
    cond2rhs = sympy.sympify('20')
    cond2 = LinIneq(cond2lhs, Op('<='), cond2rhs)

    cond3lhs = sympy.sympify("2*x1 + 2*x2 + x3")
    cond3rhs = sympy.sympify("20")
    cond3 = LinIneq(cond3lhs, Op('<='), cond3rhs)

    cost = sympy.sympify('-10*x1 - 12*x2 - 12*x3') # NOTE: if updating, update costvec too 

    lin = Linear()
    lin.add_vars(varname_syms) 
    lin.add_constraint(cond1, svgen=global_svgen)
    lin.add_constraint(cond2, svgen=global_svgen)
    lin.add_constraint(cond3, svgen=global_svgen)

    for v in varname_syms:
        lin.gez.append(v) 

    # prep to solve 
    lin.eqz_to_Ab() 
    lin.clean_Ab()

    # make cost evaluator vector 
    costvec = np.array([-10, -12, -12, 0, 0, 0], dtype=np.float32) 

    # start solving - get first BFS 
    x0, inds0 = lin.get_bfs0()
    c0 = cost.subs(lin.x_to_subsdict(x0)) # we didn't actually use this oops

    print("FIRST BFS FOUND! VALUES:")
    print("X0", x0)
    print("INDS0", inds0)
    print("COST", np.dot(costvec, x0))
    print() 
    

    while True: # while haven't found solution yet
    #for _ in range(2): 

        # calculate change in cost for each direction
        final_dcost, final_inidx, final_outidx, final_dx = None, None, None, None 
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
            B = lin.A[:,inds0]
            drest = - np.linalg.inv(B) @ db
            # we only need to consider drest < 0
            # NOTE TO SELF: if i decide to consider all again, remember the corner case
            # about x0[inds0][?]=0 but drest>0 so it's actually inf, but it's 0 in the dist calculation

            dists = np.where(drest<0, x0[inds0] / -drest, np.Inf)
            awayidx = np.argmin(dists) 
            

            if dists[awayidx] == np.Inf:
                # that means that cost can be negative infinity
                print("COST CAN BE NEGATIVE INFINITY!!")
                1/0 # TODO

            #print("DISTS:", dists) 


            # get change in cost after moving - increase db component and drest component 
            # (dists[awayidx] * (db+)
            dx = np.zeros_like(lin.A[0])
            dx[inds0] = dists[awayidx] * drest
            dx[dirn_no] = dists[awayidx] # * 1 
            dcost = np.dot( costvec , dx )

            if print_every_pivot_dirn: 
                # debug 
                print("PIVOT DIRN", dirn_no)
                print("DX:", dx)
                print("DCOST:", dcost)
                print("DISTS:", dists)
                print("DREST:", drest)
                print() 

            # check to make sure the >=0 constraints will still be fulfilled 

            # if cost ~doesnt increase~ stricly decreases
            # what about basis changes? 
            if dcost < -EPS: #= EPS:
                # convert awayidx to the normal x coordinates
                away_idx = inds0[awayidx]
                final_dcost = dcost
                final_inidx = dirn_no
                final_outidx = away_idx
                final_dx = dx
                break # found this, since it's lowest index, take it 
            
                #dcosts[dcost] = (dirn_no, away_idx, dx) # save this direction 

        if final_dcost is None: 
            # optimal already, at a local (global) minimum
            print("OPTIMIZED YAY")
            1/0

        # replace
        outidx_idx = inds0.index(final_outidx)
        inds0[outidx_idx] = final_inidx
        x0 += final_dx

        # should be done. Can wait for next loop 
        if print_every_new_bfs:
            print("LOOP DONE! NEW VALUES:")
            print("X0", x0)
            print("INDS0", inds0)
            print("COST", np.dot(costvec, x0))
            print()
        
