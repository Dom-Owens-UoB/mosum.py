import numpy as np
import sys
import warnings

import pandas as pd

from mosum.mosum import mosum
from mosum.exhaust_bic import exhaust_sc, extract_sub, get_comb_ind
from mosum.bandwidth import bandwidths_default, multiscale_grid
from mosum.mosum_test import pValue
from mosum.classes import multiscale_cpts_lp
from mosum.bootstrap import confint_multiscale_cpts



# Define function
def multiscale_localPrune(x, G=None, max_unbalance=4,
                          threshold='critical_value', alpha=.1, threshold_function=None,
                          criterion='eta', eta=.4, epsilon=.2,
                          rule='pval', penalty='log', pen_exp=1.01,
                          do_confint=False, level=.05, N_reps=1000):
    """
     Multiscale MOSUM algorithm with localised pruning

     Parameters
     ----------
     x : list
         input data
     G : int
         vector of bandwidths; given as either integers less than `len(x)/2`,
          or numbers between `0` and `0.5` describing the moving sum bandwidths relative to `len(x)`
     max_unbalance : float
        a numeric value for the maximal ratio between maximal and minimal bandwidths to be used for candidate generation,
        at least 1
     threshold : Str
         indicates which threshold should be used to determine significance.
         By default, it is chosen from the asymptotic distribution at the given significance level 'alpha`.
         Alternatively it is possible to parse a user-defined function with 'threshold_function'.
     alpha : float
         numeric value for the significance level with '0 <= alpha <= 1';
         use iff 'threshold = "critical_value"'
     threshold_function : function
         criterion : Str
        indicates how to determine whether each point 'k' at which MOSUM statistic
        exceeds the threshold is a change point; possible values are
        'eta' : there is no larger exceeding in an 'eta*G' environment of 'k'
        'epsilon' : 'k' is the maximum of its local exceeding environment, which has at least size 'epsilon*G'
    eta : float
        a positive numeric value for the minimal mutual distance of changes,
        relative to moving sum bandwidth (iff 'criterion = "eta"')
    epsilon : float
        a numeric value in (0,1] for the minimal size of exceeding environments,
        relative to moving sum bandwidth (iff 'criterion = "epsilon"')
    rule : Str
        Choice of sorting criterion for change point candidates in merging step.
        Possible values are:
        'pval' : smallest p-value
        'jump' : largest (rescaled) jump size
    penalty : Str
        Type of penalty term to be used in Schwarz criterion; possible values are:
        'log' : use 'penalty = log(len(x))**pen_exp'
        'polynomial' : use 'penalty = len(x)**pen_exp'
    pen_exp : float
        penalty exponent
    do_confint : bool
         flag indicating whether to compute the confidence intervals for change points
    level : float
         use iff 'do_confint = True'; a numeric value ('0 <= level <= 1') with which '100(1-level)%'
         confidence interval is generated
    N_reps : int
         use iff 'do.confint = True'; number of bootstrap replicates to be generated

     Returns
     -------
     multiscale_cpts object containing
     x : list
         input data
     G : int
         bandwidth vector
     threshold, alpha, threshold_function, eta
         input
     cpts : ndarray
         estimated change point
     cpts_info : DataFrame
         information on change points, including detection bandwidths, asymptotic p-values, scaled jump sizes
     pooled_cpts : ndarray
         change point candidates
     do_confint : bool
         input
     ci
         confidence intervals

     Examples
     --------
     >>> import mosum
     >>> xx = mosum.testData("mix")["x"]
     >>> xx_m  = mosum.multiscale_localPrune(xx, G = [8,15,30,70])
     >>> xx_m.summary()
     >>> xx_m.print()
     """
    n = len(x)

    if G is None:
        G = bandwidths_default(n, G_min=max(20, np.ceil(0.05 * n)))
        grid = multiscale_grid(G, max_unbalance=max_unbalance)
    elif type(G) in [int, float]:
        grid = multiscale_grid([G], max_unbalance=max_unbalance)
    elif type(G) == 'multiscale_grid_obj':
        grid = G
    elif type(G) == list:
        G.sort()
        grid = multiscale_grid(G, max_unbalance=max_unbalance)
    else: sys.exit('Expecting a vector of numbers')

    abs_bandwidth = (np.array(grid.grid) >= 1).all()

    assert max_unbalance >= 1
    assert rule in ['pval', 'jump', 'lr', 'rl']
    assert criterion in ['eta', 'epsilon']
    assert (criterion == 'eta' and 0 < eta <= 1) or (criterion == 'epsilon' and 0 < epsilon <= 1)
    assert not do_confint or N_reps > 0

    if penalty == 'log':
        log_penalty = True
    else:
        if penalty != 'polynomial':
            raise ValueError('penalty has to set to log or polynomial')
        log_penalty = False

    if threshold not in ('critical_value', 'custom'):
        raise ValueError("threshold must be either 'critical_value' or 'custom'")

    all_cpts0 = np.empty((0, 6))
    for i in range(len(grid.grid[0])):
        G1 = int(grid.grid[0][i])
        G2 = int(grid.grid[1][i])
        if threshold == 'critical_value':
            m = mosum(x, G=G1, G_right=G2,
                                 threshold='critical_value', alpha=alpha,
                                 criterion=criterion, eta=eta, epsilon=epsilon)
        else:
            threshold_val = threshold_function(G1, G2, n, alpha)
            m = mosum(x, G=G1, G_right=G2,
                                 threshold='custom', threshold_custom=threshold_val, alpha=alpha,
                                 criterion=criterion, eta=eta, epsilon=epsilon)

        if len(m.cpts) > 0:
            if not abs_bandwidth:
                G1 = int(np.floor(G1*n))
                G2 = int(np.floor(G2*n))
            all_cpts0 = np.vstack((all_cpts0,
                                   np.stack([m.cpts, np.repeat(G1, len(m.cpts)), np.repeat(G2, len(m.cpts)), np.repeat(G1+G2, len(m.cpts)),
                                 pValue(m.stat[m.cpts], n, G1, G2), m.stat[m.cpts] * np.sqrt(G1+G2) / np.sqrt(G1*G2)], axis = 1)))


    all_cpts0 = all_cpts0[np.argsort(all_cpts0[:, 0]), :]
    all_cpts = dup_merge(all_cpts0)
    ac = all_cpts.shape[0]
    if ac > 0:
        lp = local_prune(x, all_cpts, rule, log_penalty, pen_exp)
        est_cpts = lp['est_cpts']
        est_cpts_ind = detect_interval(all_cpts0, est_cpts)
        min_cost = lp['min_cost']
    else:
        est_cpts_ind = est_cpts = np.array([])
        min_cost = np.sum(x ** 2) - n * np.mean(x) ** 2

    if len(est_cpts_ind)>0:
        est_cpts_info = pd.DataFrame({
            'cpts': all_cpts0[est_cpts_ind, 0].astype(int),
            'G_left': all_cpts0[est_cpts_ind, 1].astype(int),
            'G_right': all_cpts0[est_cpts_ind, 2].astype(int),
            'p_value': all_cpts0[est_cpts_ind, 4],
            'jump': all_cpts0[est_cpts_ind, 5]
        })
    else:
        est_cpts_info = pd.DataFrame({
            'cpts': [],
            'G_left': [],
            'G_right': [],
            'p_value': [],
            'jump': []
        })

    if log_penalty:
        penalty_term = len(est_cpts) * np.log(n) ** pen_exp
    else:
        penalty_term = len(est_cpts) * n ** pen_exp

    final_sc = n / 2 * np.log(min_cost / n) + penalty_term

    if not abs_bandwidth:
        G = np.floor(n * np.sort(np.unique(np.concatenate(grid['grid']))))


    out = multiscale_cpts_lp(x, est_cpts, est_cpts_info, all_cpts[:, 0], G,
                             alpha, threshold, threshold_function, criterion, eta,
                             epsilon, final_sc, rule, penalty, pen_exp, False, None)
    if do_confint:
        out.ci = confint_multiscale_cpts(out, level=level, N_reps=N_reps)
        out.do_confint = True

    return out


def local_prune(x, all_cpts, rule, log_penalty, pen_exp):
    THRESH_MANUAL_MERGING = 24

    n = len(x)
    ac = all_cpts.shape[0]
    all_cpts = np.hstack([all_cpts, np.arange(1, ac + 1).reshape(-1, 1)])
    #all_cpts = all_cpts.astype(int)
    cand_used = np.repeat(False, ac)
    all_unique_cpts = np.hstack([-1, all_cpts[:, 0], n-1])
    all_unique_cpts = all_unique_cpts.astype(int)
    auc = len(all_unique_cpts) - 2
    sums = np.zeros((auc + 1, 4), dtype = int)
    for j in range(auc + 1):
        sums[j, 0] = all_unique_cpts[j]+1
        sums[j, 1] = all_unique_cpts[j+1]+1
        sums[j, 2] = np.sum(x[sums[j, 0]:sums[j, 1]])
        sums[j, 3] = np.sum(x[sums[j, 0]:sums[j, 1]] ** 2)
    min_cost = np.sum(sums[:, 3] - sums[:, 2] ** 2 / (sums[:, 1] - sums[:, 0]))

    current = pool = np.arange(ac)
    if rule == 'pval':
        u = all_cpts[np.lexsort((all_cpts[:, 4], all_cpts[:, 3], all_cpts[:, 1], all_cpts[:, 2])), :]
        rule_seq = u[:, 6].astype(int)
    if rule == 'jump':
        u = all_cpts[np.lexsort((-all_cpts[:, 5], all_cpts[:, 3], all_cpts[:, 1], all_cpts[:, 2])), :]
        rule_seq = u[:, 6].astype(int)
    if rule == 'lr':
        rule_seq = pool
    if rule == 'rl':
        rule_seq = np.flip(pool)

    est_cpts_ind = est_cpts = np.array([])
    while len(pool) > 0:
        j = rule_seq[0]
        adj = 0
        le = local_env(j, est_cpts_ind, all_cpts, current, ac)
        li = le['li']
        li_final = le['li_final']
        ri = le['ri']
        ri_final = le['ri_final']
        left = li + 1
        right = ri - 1
        cand_ind = np.intersect1d(np.arange(left, right + 1), pool, assume_unique=True)
        cand = all_cpts[cand_ind,0]
        ind_middl_tmp = sums[li +1:ri,1]
        ind_middl_tmp = ind_middl_tmp[np.where(~cand_used[li + 1:ri])]
        ind_tmp = np.hstack([sums[li + 1,0], ind_middl_tmp, sums[ri,1]])
        sub_sums = extract_sub(ind_tmp, x)
        doExhaustiveSearch = True
        if len(cand) > THRESH_MANUAL_MERGING:
            # Count neighbourhood size of neighbours
            cand_rule_seq = rule_seq[np.isin(rule_seq, cand_ind)]
            cand_size = np.empty(len(cand), dtype=int)
            cand_size[0] = len(cand)
            for i_tmp in range(1, len(cand)):
                jj = cand_rule_seq[i_tmp]
                le_jj = local_env(jj, est_cpts_ind, all_cpts, current, ac)
                left_jj = le_jj['li'] + 1
                right_jj = le_jj['ri'] - 1
                cand_ind_jj = np.array([ind for ind in range(left_jj, right_jj + 1) if ind in pool])
                # cand_jj <- all.cpts[cand_ind_jj, 1] # = D
                cand_size[i_tmp] = len(cand_ind_jj)

            if np.any(cand_size <= THRESH_MANUAL_MERGING):
                # Proceed with next candidate, for which exhaustive search IS possible
                rule_tmp = cand_rule_seq[np.argmin(cand_size <= THRESH_MANUAL_MERGING)]
                ind_star = np.where(rule_seq == rule_tmp)
                rule_seq[ind_star] = rule_seq[0]
                rule_seq[0] = rule_tmp
                doExhaustiveSearch = False
            else:
                # Count neighbourhood size of remaining candidates
                cand_size = np.empty(len(rule_seq))
                cand_size[0] = len(cand)
                for i_tmp in range(1, len(rule_seq) - 1):
                    jj = rule_seq[i_tmp]
                    le_jj = local_env(jj, est_cpts_ind, all_cpts, current, ac)
                    left_jj = le_jj['li'] + 1
                    right_jj = le_jj['ri'] - 1
                    cand_ind_jj = np.array([ind for ind in range(left_jj, right_jj + 1) if ind in pool])
                    # cand_jj <- all.cpts[cand_ind_jj, 1] # = D
                    cand_size[i_tmp] = len(cand_ind_jj)

                if np.any(cand_size <= THRESH_MANUAL_MERGING):
                    # Proceed with next candidate, for which exhaustive search IS possible
                    ind_star = np.argmin(cand_size <= THRESH_MANUAL_MERGING)
                    rule_tmp = rule_seq[ind_star]
                    rule_seq[ind_star] = rule_seq[0]
                    rule_seq[0] = rule_tmp
                    doExhaustiveSearch = False
                else:
                    # No more exhaustive search possible at all
                    # --> Do manual merging, until exhaustive search becomes possible
                    while len(cand) > THRESH_MANUAL_MERGING:
                        warn_msg = 'Warning: ' + str(len(cand)) + ' conflicting candidates, thinning manually'
                        warnings.warn(warn_msg)
                        k = cand[np.argmin(np.diff(cand))]
                        l = np.where(sub_sums[:, 1] == k)[0][0]
                        a, b = sub_sums[l], sub_sums[l + 1]
                        adj = adj + ((a[1] - b[1] + 1) * (a[2] - a[1] + 1) / (b[2] - b[1] + 1) * (
                                    a[3] / (a[2] - a[1] + 1) - b[3] / (b[2] - b[1] + 1)) ** 2)
                        sub_sums[l + 1, 0] = a[0]
                        sub_sums[l + 1, 2:4] = sub_sums[l + 1, 2:4] + a[2:4]
                        sub_sums = np.delete(sub_sums, l, axis=0)
                        cand = np.setdiff1d(cand, k)
                        k_ind = np.where(all_cpts[:, 0] == k)[0][0]
                        cand_ind = np.setdiff1d(cand_ind, k_ind)
                        pool = np.setdiff1d(pool, k_ind)
                        rule_seq = np.setdiff1d(rule_seq, k_ind)
                        cand_used[k_ind] = True
        if doExhaustiveSearch:
            # step 4
            # performs exhaustive search (Algorithm 2)
            out = exhaust_sc(cand=cand, sub_sums=sub_sums,
                             strength=pen_exp, log_penalty=log_penalty,
                             n=n, auc=len(current), min_cost=min_cost)
            est_cpts = np.append(est_cpts, out['est_cpts'])
            current_est_cpts_ind = all_cpts[np.isin(all_cpts[:, 0], out['est_cpts']), 6].astype(int)
            est_cpts_ind = np.append(est_cpts_ind, current_est_cpts_ind)

            # steps 5, 6
            # removal of candidates
            rm_set = np.append(j, current_est_cpts_ind)-1
            if len(current_est_cpts_ind) > 0:
                rm_set = np.append(rm_set, cand_ind[(cand_ind >= np.min(current_est_cpts_ind)-1) &
                                                    (cand_ind <= np.max(current_est_cpts_ind)-1 )])
                if li_final:
                    rm_set = np.append(rm_set, cand_ind[cand_ind <= np.max(current_est_cpts_ind)-1])
                if ri_final:
                    rm_set = np.append(rm_set, cand_ind[cand_ind >= np.min(current_est_cpts_ind)-1])

            pool = np.setdiff1d(pool, rm_set)
            cand_used[np.unique(rm_set)] = True
            rule_seq = np.setdiff1d(rule_seq-1, rm_set)+1
            current = np.append(pool, est_cpts_ind)
            current = current.astype(int)
            #if(len(current))
            current_cands = np.isin(cand, all_cpts[current-1, 0])
            ind_star = get_comb_ind(current_cands)
            min_cost = min_cost + adj - out['sc'][-1, 0] + out['sc'][ind_star, 0]

    est_cpts = np.sort(np.ravel(est_cpts));
    est_cpts_ind = np.sort(np.ravel(est_cpts_ind))

    return {'est_cpts': est_cpts.astype(int), 'est_cpts_ind': est_cpts_ind.astype(int), 'min_cost': min_cost}


def local_env(j, est_cpts_ind, all_cpts, current, ac):
    li_final, ri_final = True, True
    if sum(est_cpts_ind < j):
        li = int(max(est_cpts_ind[est_cpts_ind < j]))-1
    else:
        li = -1
    if j > 1:
        ind = [i for i in range(li+1, j) if i in current]
        tmp = (all_cpts[j-1, 0] - all_cpts[ind, 0]) >= np.maximum(all_cpts[j-1, 1], all_cpts[ind, 2])
        if len(ind) > 0 and tmp.any():
            li_tmp = np.array(ind)[tmp].max()
        else:
            li_tmp = li
        if li_tmp > li:
            li = li_tmp
            li_final = False

    if sum(est_cpts_ind > j):
        ri = int(min(est_cpts_ind[est_cpts_ind > j]))
    else:
        ri = ac
    if j < ac:
        ind = [i for i in range(j, ac) if i in current]
        tmp = (all_cpts[ind, 0] - all_cpts[j-1, 0]) >= np.maximum(all_cpts[j-1, 2], all_cpts[ind, 1])
        if len(ind) > 0 and tmp.any():
            ri_tmp = np.array(ind)[tmp].min()
        else:
            ri_tmp = ri
        if ri_tmp < ri:
            ri = ri_tmp
            ri_final = False

    return {'li': li, 'li_final': li_final, 'ri': ri, 'ri_final': ri_final}

def dup_merge(all_cpts):
    all_unique_cpts = np.unique(all_cpts[:, 0], axis=0)
    out = np.empty((0, all_cpts.shape[1]))
    for k in all_unique_cpts:
        ind = np.where(all_cpts[:, 0] == k)[0]
        ind_min = ind[all_cpts[ind, 3].argmin()]
        if ind_min.size > 1:
            ind_min = ind_min[np.argmin(all_cpts[ind_min, 3])]
        out = np.vstack((out, all_cpts[ind_min, :]))
    return out

def detect_interval(all_cpts, est_cpts):
    new_est_cpts_ind = []
    for k in est_cpts:
        ind = np.where(all_cpts[:, 0] == k)[0]
        ind_min = np.array(ind[all_cpts[ind, 3].argmin()])
        if ind_min.size > 1:
            ind_min = np.array(ind_min[all_cpts[ind, 4].argmin()])
            if ind_min.size >1:
                ind_min = ind_min[all_cpts[ind, 5].argmax()]
        new_est_cpts_ind.append(int(ind_min))
    return new_est_cpts_ind

