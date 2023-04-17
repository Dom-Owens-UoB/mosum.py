import numpy as np
import math
#from numba import jit

# Get integer vector of changepoint indices,
# based on bool-field representation of combinations.
# E.g. for combination index 11 [=1011]: get_comb_ind([True, False, True, True])=11
def get_comb_ind(active):
    m = len(active)
    res = 0
    for j in range(m):
        res += active[j] * (1 << j)
    return res


# Helping function for algorithm 2: Pre-compute the partial sums
# S_i = sum{j=k_i+1}^{k_{i+1}}x_i and the partial sums of squared
# T_i = sum{j=k_i+1}^{k_{i+1}}x_i^2
# between the (sorted) candidates k_i and k_{i+1} in cand.
# Output: data frame with 4 columns k_i | k_{i+1} | S_i | T_i
def extract_sub(cand, x):
    m = len(cand)
    res = np.zeros((m - 1, 4))
    i = 0  # position in cand vector
    j = cand[i]  # position in x vector
    sum_val = 0.0
    sum_sq_val = 0.0
    while i + 1 < m:
        sum_val += x[j]
        sum_sq_val += x[j] ** 2
        if j + 1 == cand[i + 1]:
            res[i, 0] = cand[i] + 1
            res[i, 1] = cand[i + 1]
            res[i, 2] = sum_val
            res[i, 3] = sum_sq_val
            sum_val = 0.0
            sum_sq_val = 0.0
            i += 1
        j += 1
    return res


# Starting value to iterate (in lexicographical order)
# over all bit permutaions having l bits set to 1.
# E.g.: start_bit_permutations(2) = 3 [=0..011].
def start_bit_permutations(l):
    return (1 << l) - 1


# Next value to iterate (in lexicographical order) over all bit
# permutaions having l bits set to 1.
# Example sequence (2 bits): {0011, 0101, 0110, 1001, 1010, 1100}.
# Source: https://stackoverflow.com/questions/1851134/generate-all-binary-strings-of-length-n-with-k-bits-set
def next_bit_permutation(v):
    t = (v | (v - 1)) + 1
    w = t | ((((t & -t) // (v & -v)) >> 1) - 1)
    return w


# Is index i_child a child of index i_parent?
# ASSERT: i_child is of the form (i_parent XOR i_help),
#         with i_help having exactly one non-zero bit
def is_child(i_child, i_parent):
    return i_child < i_parent


def numberOfSetBits(i: int) -> int:
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24

def comb_contains_cpt(comb: int, k_ind: int) -> bool:
    return comb & (1 << k_ind)


def get_local_costs(icomb: int, sub_sums: np.ndarray) -> float:
    m = sub_sums.shape[0] - 1
    res = 0.0
    A = 0.0
    B = 0.0
    C = 0.0
    for j in range(m + 1):
        A += sub_sums[j, 3]
        B += sub_sums[j, 2]
        C += sub_sums[j, 1] - sub_sums[j, 0] + 1.0
        if j == m or comb_contains_cpt(icomb, j):
            res += A - B * B / C
            A = 0.0
            B = 0.0
            C = 0.0
    return res


def setBitNumber(n: int) -> int:
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n = n + 1
    return (n >> 1)

#@jit
def exhaust_sc(cand, sub_sums, strength, log_penalty, n, auc, min_cost):
    m = len(cand)
    M = (1 << m)
    n_half = n / 2.0

    sc_penalty = math.pow(math.log(n) if log_penalty else n, strength)

    INF = float('inf')

    flag = [True] * M
    sc_vals = [INF] * M
    cost_vals = [INF] * M  # local costs
    num_cpts = [np.nan] * M
    final = []
    m_star = m

    min_cost_local = np.sum(
        sub_sums[:, 3] - sub_sums[:, 2] ** 2 / (sub_sums[:, 1] - sub_sums[:, 0]+1)) #+ 1.0
    # M-1 [=1...1] represents ALL changes
    # 0 [=0...0] represents NO change
    cost_vals[M - 1] = min_cost_local
    sc_vals[M - 1] = n_half * np.log(min_cost / n) + auc * sc_penalty
    num_cpts[M - 1] = m

    # iterate over all combination lengths
    l = m
    while l > 0:
        # iterate over all combinations of length l
        # step 1: pruning: inherit FALSE flags in next generation
        i_parent = start_bit_permutations(l)
        count = 0
        while i_parent < M:
            if not flag[i_parent]:
                for i_child_help in range(m):
                    i_child = i_parent ^ (1 << i_child_help)
                    if flag[i_child]:
                        if is_child(i_child, i_parent):
                            flag[i_child] = False
            else:
                count += 1
                final.append(i_parent)
                if m_star > l:
                    m_star = l
            i_parent = next_bit_permutation(i_parent)
        if count == 0:
            break

        # iterate over all combinations of length l
        # step 2: Compute SCs and update flags
        i_parent = start_bit_permutations(l)
        while i_parent < M:
            if flag[i_parent]:
                for i_child_help in range(m):
                    i_child = i_parent ^ (1 << i_child_help)
                    if is_child(i_child, i_parent) and flag[i_child]:
                        # step 2.1: compute SC
                        if cost_vals[i_child] == INF:
                            cost_vals[i_child] = get_local_costs(i_child, sub_sums)
                            child_cost = min_cost - min_cost_local + cost_vals[i_child]
                            num_cpts[i_child] = l - 1
                            child_auc = auc - m + num_cpts[i_child]
                            sc_vals[i_child] = n_half * math.log(child_cost / n) + child_auc * sc_penalty
                        # step 2.2: pruning
                        if sc_vals[i_parent] < sc_vals[i_child]:
                            flag[i_child] = False
            i_parent = next_bit_permutation(i_parent)
        l -= 1

    if cost_vals[0] != INF and flag[0]:
        final.append(0)
        m_star = 0

    # index of final combination (minimize sc, if several)
    final_ind_star = 0
    min_sc = INF
    left, right, j, jj = 0, 0, 0, 0
    for i in range(len(final)):
        j = final[i]
        if (num_cpts[j] >= m_star) and (num_cpts[j] <= m_star + 2):
            if sc_vals[j] < min_sc:
                min_sc = sc_vals[j]
                final_ind_star = j
            if num_cpts[j] >= 1:
                left = setBitNumber(j)
                if num_cpts[j] >= 2:
                    right = j ^ (j & (j - 1))
                else:
                    right = left
                jj = j - left
                if cost_vals[jj] == INF:
                    cost_vals[jj] = get_local_costs(jj, sub_sums)
                    child_cost = min_cost - min_cost_local + cost_vals[jj]
                    num_cpts[jj] = num_cpts[j] - 1
                    child_auc = auc - m + num_cpts[jj]
                    sc_vals[jj] = n_half * np.log(child_cost / float(n)) + child_auc * sc_penalty
                if sc_vals[jj] < min_sc:
                    min_sc = sc_vals[jj]
                    final_ind_star = jj
                if num_cpts[j] >= 2:
                    jj = j - right
                    if cost_vals[jj] == INF:
                        cost_vals[jj] = get_local_costs(jj, sub_sums)
                        child_cost = min_cost - min_cost_local + cost_vals[jj]
                        num_cpts[jj] = num_cpts[j] - 1
                        child_auc = auc - m + num_cpts[jj]
                        sc_vals[jj] = n_half * np.log(child_cost / float(n)) + child_auc * sc_penalty
                    if sc_vals[jj] < min_sc:
                        min_sc = sc_vals[jj]
                        final_ind_star = jj
                    jj = j - left - right
                    if cost_vals[jj] == INF:
                        cost_vals[jj] = get_local_costs(jj, sub_sums)
                        child_cost = min_cost - min_cost_local + cost_vals[jj]
                        num_cpts[jj] = num_cpts[j] - 2
                        child_auc = auc - m + num_cpts[jj]
                        sc_vals[jj] = n_half * np.log(child_cost / float(n)) + child_auc * sc_penalty
                    if sc_vals[jj] < min_sc:
                        min_sc = sc_vals[jj]
                        final_ind_star = jj

    # get estimated changepoints, according to final combination
    est_cpts = []
    est_cpts_ind = 0
    for j in range(m):
        if comb_contains_cpt(final_ind_star, j):
            est_cpts.append(cand[j])
            est_cpts_ind += 1

    sc = np.zeros((M, 2))
    sc[:, 0] = cost_vals
    sc[:, 1] = sc_vals

    res = {}
    res["sc"] = sc
    res["est_cpts"] = est_cpts
    res["final"] = final_ind_star
    res["num_cpts"] = num_cpts
    res["finals"] = final

    return res
