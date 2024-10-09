import sympy 
import numpy as np 
EPS = 1e-10 

from linopt import Ops, Op, LinIneq, Linear, global_svgen, global_varnames 


if __name__=='__main__':
    print_every_pivot_dirn = False
    print_every_new_bfs = True 

    # x_{}{}{}o must be defined in other way 
    def get_overall(team, role, num): 
        return 70 
    
    # gk scores 
    gkA = 90 
    gkB = 80 

    varname_syms = [] 
    for team in ['A', 'B']: 
        for role in ['d', 'm', 'a']: # defender midfielder attacker
            max_num = 4 if role=='a' else 3 
            for num in range(1, 1+max_num): 
                # shooting, passing, defending 
                varname_syms.append(sympy.Symbol("x_{}{}{}s".format(team, role, num))) 
                varname_syms.append(sympy.Symbol("x_{}{}{}p".format(team, role, num))) 
                varname_syms.append(sympy.Symbol("x_{}{}{}d".format(team, role, num))) 

        # settle gk separately - actually, it shouldn't be a variable 
        #varname_syms.append(sympy.Symbol('x_{}g'.format(team))) 
    
    # introduce n and scores for each team 
    varname_syms.append(sympy.Symbol('n')) 
    varname_syms.append(sympy.Symbol('Ascore'))
    varname_syms.append(sympy.Symbol('Bscore'))


    for v in varname_syms:
        global_varnames.append(v)

    
    overall_conds = [] # overall score constraints 
    max_conds = [] # max for each score (min is handled separately )
    for team in ['A', 'B']: 
        for role in ['d', 'm', 'a']: # defender midfielder attacker
            max_num = 4 if role=='a' else 3 
            for num in range(1, 1+max_num): 
                s = "x_{}{}{}s".format(team, role, num) 
                p = "x_{}{}{}p".format(team, role, num) 
                d = "x_{}{}{}d".format(team, role, num)
                condlhs = sympy.sympify("{} + {} + {}".format(s, p, d)) 
                condrhs = sympy.sympify(str( get_overall(team, role, num)*3 )) 
                overall_conds.append(LinIneq(condlhs, Op('='), condrhs)) 

                max_conds.append(LinIneq(sympy.sympify(s), Op('<='), sympy.sympify('99'))) 
                max_conds.append(LinIneq(sympy.sympify(p), Op('<='), sympy.sympify('99'))) 
                max_conds.append(LinIneq(sympy.sympify(d), Op('<='), sympy.sympify('99')))     
    

    # n - all 8 cases for approximating ------------------------------------- 

    # possesson times 
    tAstr = "(3/10)*(" 
    tAvars = [] 
    for role in ['d', 'm', 'a']: # defender midfielder attacker
        max_num = 4 if role=='a' else 3 
        for num in range(1, 1+max_num): 
            tAvars.append("x_A{}{}p".format(role, num)) 
    tAstr += '+'.join(tAvars) + ')' 

    tBstr = "(3/10)*(" 
    tBvars = [] 
    for role in ['d', 'm', 'a']: # defender midfielder attacker
        max_num = 4 if role=='a' else 3 
        for num in range(1, 1+max_num): 
            tBvars.append("x_B{}{}p".format(role, num)) 
    tBstr += '+'.join(tBvars) + ')' 

    tAtB = [('t_A', tAstr), ('t_B', tBstr)] 

    # the 8 approxes 
    lhs1 = sympy.sympify('1.6 - 0.007*(t_A + t_B)') 
    lhs1 = lhs1.subs(tAtB).expand() 
    ncond1 = LinIneq(lhs1, Op('='), sympy.Symbol('n')) 

    lhs2 = sympy.sympify('1.05 - 0.003*(t_A + t_B)') 
    lhs2 = lhs2.subs(tAtB).expand() 
    ncond2 = LinIneq(lhs2, Op('='), sympy.Symbol('n')) 

    lhs3 = sympy.sympify('0.85 - 0.002*(t_A + t_B)') 
    lhs3 = lhs3.subs(tAtB).expand() 
    ncond3 = LinIneq(lhs3, Op('='), sympy.Symbol('n')) 

    lhs4 = sympy.sympify('0.6 - 0.001*(t_A + t_B)') 
    lhs4 = lhs4.subs(tAtB).expand() 
    ncond4 = LinIneq(lhs4, Op('='), sympy.Symbol('n')) 

    lhs5 = sympy.sympify('0.53 - 0.0008*(t_A + t_B)') 
    lhs5 = lhs5.subs(tAtB).expand() 
    ncond5 = LinIneq(lhs5, Op('='), sympy.Symbol('n')) 

    lhs6 = sympy.sympify('0.43 - 0.0005*(t_A + t_B)') 
    lhs6 = lhs6.subs(tAtB).expand() 
    ncond6 = LinIneq(lhs6, Op('='), sympy.Symbol('n')) 

    lhs7 = sympy.sympify('0.27 - 0.0002*(t_A + t_B)') 
    lhs7 = lhs7.subs(tAtB).expand() 
    ncond7 = LinIneq(lhs7, Op('='), sympy.Symbol('n')) 

    lhs8 = sympy.sympify('0.19 - 0.0001*(t_A + t_B)') 
    lhs8 = lhs8.subs(tAtB).expand() 
    ncond8 = LinIneq(lhs8, Op('='), sympy.Symbol('n')) 

    # end n -----------------------------------------------------------------

    # points 
    avgdefA = sympy.sympify("(1/3)*(x_Ad1d + x_Ad2d + x_Ad3d)").expand() 
    shootingA = sympy.sympify("(1/4)*(x_Aa1s + x_Aa2s + x_Aa3s + x_Aa4s)")
    overallAs = [] 
    for role in ['d', 'm', 'a']: # defender midfielder attacker
        max_num = 4 if role=='a' else 3 
        for num in range(1, 1+max_num): 
            overallAs.append(get_overall("A", role, num)) 
    avgoverallA = np.mean(overallAs) 

    avgdefB = sympy.sympify("(1/3)*(x_Bd1d + x_Bd2d + x_Bd3d)").expand() 
    shootingB = sympy.sympify("(1/4)*(x_Ba1s + x_Ba2s + x_Ba3s + x_Ba4s)")
    overallBs = [] 
    for role in ['d', 'm', 'a']: # defender midfielder attacker
        max_num = 4 if role=='a' else 3 
        for num in range(1, 1+max_num): 
            overallBs.append(get_overall("B", role, num)) 
    avgoverallB = np.mean(overallBs) 


    avg_defender_mul= ((1/10)*0 + (3/10)*0.25 + (6/10)*0.5) 

    scoreA = (0.8*shootingA - 0.05*gkA - avg_defender_mul*avgdefB) * (0.7 * avgoverallA) 
    scoreA = scoreA.expand() 
    scoreB = (0.8*shootingB - 0.05*gkB - avg_defender_mul*avgdefA) * (0.7 * avgoverallB) 
    scoreB = scoreB.expand() 

    scoreconds = [LinIneq(scoreA, Op('='), sympy.Symbol('Ascore')), 
                  LinIneq(scoreB, Op('='), sympy.Symbol('Bscore')), ] 


    # to maximize A's score 
    cost = sympy.sympify('Bscore - Ascore') # NOTE: if updating, update costvec too 



    # SPLIT INTO THE 8 DIFFERENT RANGES OF VALUES TO YES 

    for nrange in range(1, 9):

        # define the solver 
        lin = Linear()
        lin.add_vars(varname_syms) 


        # add the constraints 
        # add this constraint 
        exec("lin.add_constraint(ncond{}, svgen=global_svgen)".format(nrange), locals())
        # only a range of values of n 
        if nrange > 1: 
            exec("lin.add_constraint(LinIneq(lhs{}, '<=', lhs{}), svgen=global_svgen)".format(nrange-1, nrange), locals())
        if nrange < 8: 
            exec("lin.add_constraint(LinIneq(lhs{}, '<=', lhs{}), svgen=global_svgen)".format(nrange+1, nrange), locals())

        print("DOING FOR RANGE OF VALUES WHERE", 
              eval("lhs{}".format(nrange), locals()), 
              "IS THE HIGHEST") 

        for cond in overall_conds: 
            lin.add_constraint(cond, global_svgen) 
        for cond in max_conds: 
            lin.add_constraint(cond, global_svgen) 
        for cond in scoreconds: 
            lin.add_constraint(cond, global_svgen) 


        # each score is at least 1 
        for v in varname_syms:
            lin.gez.append(v) 

        # prep to solve 
        lin.eqz_to_Ab() 
        lin.clean_Ab()

        # make cost evaluator vector 
        costvec = [1 if str(v)=='Bscore' else (-1 if str(v) == 'Ascore' else 0) for v in lin.varnames] 
        costvec = np.array(costvec, dtype=np.float32) 


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
            
