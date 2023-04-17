import numpy as np
from numba import njit



def confint_mosum_cpts(object, parm: str = "cpts", level: float = 0.05, N_reps: int = 1000):
    #assert isinstance(object, classes.mosum_obj), "object must be of class mosum.mosum_obj"
    assert parm == "cpts", "parm must be 'cpts'"
    if object.do_confint:
        return object.ci
    else:
        return cpts_bootstrap(object, N_reps, level)


def confint_multiscale_cpts(object, parm: str = "cpts", level: float = 0.05, N_reps: int = 1000):
    #if not isinstance(object, classes.multiscale_cpts):
    #    raise ValueError("Input object must be of class multiscale_cpts")
    if not parm == 'cpts':
        raise ValueError("Only parm = 'cpts' is supported")
    if object.do_confint:
        return object.ci
    else:
        return cpts_bootstrap(object, N_reps, level)

def cpts_bootstrap(mcpts, N_reps, level):
    x = mcpts.x
    cpts_info = mcpts.cpts_info

    q = cpts_info["cpts"].shape[0]
    if q == 0:
        cpts, pointwise_left, pointwise_right, uniform_left, uniform_right = [], [], [], [], []
    else:
        n = len(x)
        cpts = cpts_info["cpts"]
        brks = np.concatenate(([0], cpts, [n]))
        spacing = np.diff(brks[:-1]), np.diff(brks[1:])
        cpts_info = np.column_stack((cpts_info, np.minimum(cpts_info["G_left"], np.floor(2/3 * spacing[0])),
                                            np.minimum(cpts_info["G_right"], np.floor(2/3 * spacing[1]))))
        # Get bootstrap replicates from C++
        help_out = cpts_bootstrap_help(cpts_info, x, N_reps)
        d_hat, k_star, k_star1, k_star2, sigma2_hat  = help_out["d_hat"], help_out["k_star"], help_out["k_star1"], help_out["k_star2"], help_out["sigma2_hat"]
        # Pointwise confidence intervals
        C_value_j = np.apply_along_axis(lambda v: np.quantile(np.abs(v), 1 - level), 0, k_star1)
        pointwise_left = np.ceil(np.maximum(np.maximum(1, cpts - cpts_info[:,1] + 1), cpts - C_value_j))
        pointwise_right = np.floor(np.minimum(np.minimum(n, cpts + cpts_info[:,2]), cpts + C_value_j))
        # Uniform confidence intervals
        uCI_help = np.apply_along_axis(lambda v: np.max(np.abs(v)), 1, k_star2)
        C_value = np.quantile(uCI_help, 1 - level)
        uniform_left, uniform_right = [], []
        for j in range(q):
            uniform_left.append(max(max(1, cpts[j] - cpts_info[j, 1] + 1),
                                     cpts[j] - C_value * sigma2_hat[j] / d_hat[j]**2))
            uniform_right.append(min(min(n, cpts[j] + cpts_info[j, 2]),
                                      cpts[j] + C_value * sigma2_hat[j] / d_hat[j]**2))
        uniform_left = np.ceil(uniform_left)
        uniform_right = np.floor(uniform_right)
    CI = {'cpts': cpts,
          'pw.left': pointwise_left,
          'pw.right': pointwise_right,
          'unif.left': uniform_left,
          'unif.right': uniform_right}

    return {'N_reps': N_reps,
            'level': level,
            'CI': CI}




@njit
def mean_help(x, l, r):
    """helping function for bootstrap (compute local means)"""
    if l > r:
        raise ValueError("Expecting l <= r")
    res = 0.0
    for t in range(l, r+1):
        res += x[t]
    res /= (r - l + 1)
    return res

@njit
def get_k_star(x_star, k_hat, G_l, G_r, G_ll, G_rr):
    """Compute bootstrapped mosum statistic and return maximum position thereof"""
    n = len(x_star)
    k_hat_ind = k_hat - 1
    l = max(0, k_hat_ind - G_ll + 1)
    r = min(n - 1, k_hat_ind + G_rr)
    max_val = -1.0
    max_pos = l - 1
    current_val = -1.0
    for t in range(l, r+1):
        if t < G_l - 1:
            t_val_help = 0.0
            scaling = np.sqrt((G_l + G_r) / ((t + 1) * (G_l + G_r - t - 1.0)))
            mean_l = mean_help(x_star, 0, G_l+G_r-1)
            for j in range(t+1):
                t_val_help += (mean_l - x_star[j])
            current_val = np.abs(scaling * t_val_help)
        elif t >= n - G_r:
            t_val_help = 0.0
            scaling = np.sqrt((G_l + G_r) / ((t + 1 - (n - G_l - G_r)) * (n - t - 1.0)))
            mean_r = mean_help(x_star, n - G_l - G_r, n - 1)
            for j in range(n - G_l - G_r, t+1):
                t_val_help += (mean_r - x_star[j])
            current_val = np.abs(scaling * t_val_help)
        else:
            scaling = np.sqrt(G_l * G_r / (G_l + G_r))
            mean_r = mean_help(x_star, t + 1, t + G_r)
            mean_l = mean_help(x_star, t - G_l + 1, t)
            current_val = np.abs(scaling * (mean_r - mean_l))
        if current_val > max_val:
            max_val = current_val
            max_pos = t
    return max_pos ##

@njit
def bootstrapped_timeSeries(cpts, x):
    """Obtain bootstrap replicate of time series"""
    n = len(x)
    q = len(cpts)
    x_star = np.zeros(n)
    for j in range(q+1):
        l = 0 if j==0 else cpts[j-1]
        r = n-1 if j==q else cpts[j]-1
        N = r-l+1
        x_local = x[l:r+1]
        if(N>0):
            x_star[l:r+1] = np.random.choice(x_local, size=N, replace=True)
    return x_star


def cpts_bootstrap_help(cpts_info, x, N_reps):
    """Helping function to get bootstrap replicates of change point estimates"""
    cpts = cpts_info[:,0].astype(int)
    q = len(cpts)
    n = len(x)

    k_star = np.zeros((N_reps, q), dtype=int) #bootstrapped changepoints
    k_star1 = np.zeros((N_reps, q), dtype=int) #bootstrapped differences (k^* - \hat k)
    k_star2 = np.zeros((N_reps, q), dtype=float) #bootstrapped rescaled differences d^2/sigma^2 (k^* - \hat k)

    d_hat = np.zeros(q) #estimated jump heights
    sigma2_hat = np.zeros(q) #estimated noise variance (pooled from left and right stationary part)
    for j in range(q):
        m_pos = cpts[j]-1
        l_pos = 0 if j == 0 else cpts[j-1]
        r_pos = n-1 if j == q-1 else cpts[j+1]-1
        x_l = x[l_pos:m_pos+1]
        x_r = x[m_pos+1:r_pos+1]
        d_hat[j] = np.mean(x_r) - np.mean(x_l)
        denominator = max(1.0, r_pos-l_pos-2)
        tau_l = np.var(x_l) * (len(x_l)-1) if len(x_l) > 1 else 0.0
        tau_r = np.var(x_r) * (len(x_r)-1) if len(x_r) > 1 else 0.0
        sigma2_hat[j] = (tau_l + tau_r) / denominator

    for iboot in range(N_reps):
        x_star = bootstrapped_timeSeries(cpts, x)
        for j in range(q):
            G_l = cpts_info[j, 1].astype(int)
            G_r = cpts_info[j, 2].astype(int)
            G_ll = cpts_info[j, 5].astype(int)
            G_rr = cpts_info[j, 6].astype(int)
            k_star[iboot,j] = get_k_star(x_star, cpts[j], G_l, G_r, G_ll, G_rr)
            k_star1[iboot,j] = k_star[iboot,j]-cpts[j]
            k_star2[iboot,j] = (k_star[iboot,j]-cpts[j]) * (d_hat[j]**2) / sigma2_hat[j]

    res = {"k_star": k_star,
           "k_star1": k_star1,
           "k_star2": k_star2,
           "d_hat": d_hat,
           "sigma2_hat": sigma2_hat}
    return res
