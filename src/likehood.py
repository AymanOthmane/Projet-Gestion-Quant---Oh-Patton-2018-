from src.model import *
from src.skew_t import *
from src.tools import *
import numpy as np
from scipy.interpolate import interp2d


import numpy as np

def LL_eacht_GASFacCop_Skewtt_Ngroup(theta_t, u_t,  Gcdf, Gpdf, x_grid, lam_grid, group_code, epsi):
    """
    This function computes the log likelihood of (skew t - t) factor copula evaluated at time t.
    It also generates the numerical derivatives of log copula density with respect to each group's factor loading at time t.

    Inputs:
        theta_t - A vector of parameters [factor loadings; nuinv_z; nuinv_eps; psi_z]
        u_t - A Nx1 vector Unif(0,1) that are distributed according to this copula at time t
        
        Gcdf - A [Num_x_grid by Num_lam_grid] matrix of the marginal cdfs of skew t-t factor model at x and factor loading (lam)
        Gpdf - A [Num_x_grid by Num_lam_grid] matrix of the marginal pdfs of skew t-t factor model at x and factor loading (lam)
        x_grid - A vector of x that Gcdf and Gpdf are evaluated at
        lam_grid - A vector of factor loading that Gcdf and Gpdf are evaluated at
        group_code - A Nx1 vector of group codes into which each firm is classified
        epsi - A scalar, the step size for numerical derivative for score calculation

    Outputs:
        LL_t - A scalar, the log-likelihood of (skew t - t) factor copula evaluated at each time t
        N_derivative - A (N_group x 1) vector, each element is the numerical derivative of log copula density
                        with respect to each group's factor loading
    """

    # Ensure theta_t and u_t are column vectors
    #theta_t = np.atleast_2d(theta_t).T if theta_t.shape[0] < theta_t.shape[1] else theta_t
    #u_t = np.atleast_2d(u_t).T #if u_t.shape[0] < u_t.shape[1] else u_t

    if len(group_code) != len(u_t):
        raise ValueError('The length of group_code does not equal the length of u_t')

    if len(theta_t) != (3 + max(group_code)):
        raise ValueError('theta_t or group_code has incorrect size')

    Ginv_u_t = np.full((len(u_t), 1), np.nan)
    Ginv_u_t_ateps = np.full((len(u_t), 1), np.nan)
    deno_t = np.full((len(u_t), 1), np.nan)
    deno_t_ateps = np.full((len(u_t), 1), np.nan)

    for i in range(1, max(group_code) + 1):
        inx = np.where(group_code == i)[0]
        u = u_t[inx]

        theta_group = [theta_t[i-1], theta_t[-3], theta_t[-2], theta_t[-1]]
        theta_group_eps = [theta_t[i-1] + epsi, theta_t[-3], theta_t[-2], theta_t[-1]]

        Ginv_u_t[inx, 0] = factor_cop_Gcdfinv_spline(u, theta_group, Gcdf, x_grid, lam_grid)
        Ginv_u_t_ateps[inx, 0] = factor_cop_Gcdfinv_spline(u, theta_group_eps, Gcdf, x_grid, lam_grid)

        deno_t[inx, 0] = factor_cop_Gpdfvec_Skewtt(Ginv_u_t[inx, 0], theta_group, Gpdf, x_grid, lam_grid)
        deno_t_ateps[inx, 0] = factor_cop_Gpdfvec_Skewtt(Ginv_u_t_ateps[inx, 0], theta_group_eps, Gpdf, x_grid, lam_grid)

    numer_t = factor_cop_FXpdf_GL_Skewtt(np.column_stack((Ginv_u_t, Ginv_u_t_ateps)), theta_t, group_code, epsi)

    LL_t = np.log(numer_t[0]) - np.sum(np.log(deno_t))

    # The numerical derivative of log copula density w.r.t. each group's factor loading
    N_derivative = np.full((max(group_code), 1), np.nan)
    for i in range(1, max(group_code) + 1):
        inx = np.where(group_code == i)[0]

        deno_temp = deno_t.copy()
        deno_temp[inx, 0] = deno_t_ateps[inx, 0]

        LL_eps = np.log(numer_t[i]) - np.sum(np.log(deno_temp))

        N_derivative[i-1] = (LL_eps - LL_t) / epsi

    return LL_t, N_derivative




def LL_GASFacCop_Skewtt_VT(theta, data_u, lam_bar, lam_ini):
    """
    This function computes the sum of (negative) log likelihoods of factor copula (skew t - t) with GAS recursion.

    Inputs:
        theta - A vector of parameters [alpha, beta, nuinv_z, nuinv_eps, psi_z]
        data_u - A TxN matrix, the matrix of Unif(0,1) to be modeled with this copula
        
        lam_bar - A (N x 1) vector of (average) factor loadings given from the separate stage
        lam_ini - A (N x 1) vector of factor loadings at t=1

    Outputs:
        out - A scalar, the sum of (negative) log-likelihoods of factor copula evaluated at each time t
        LL - A Tx1 vector of log-likelihoods of factor copula evaluated at each time t
        lam - A (T x N) matrix of (time-varying) factor loadings
        log_lam - A (T x N) matrix of (time-varying) log of factor loadings
        s - A (T x N) matrix of (time-varying) score
    """

    # Ensure theta is a numpy array and a column vector
    #if np.isscalar(theta):
    #    theta = np.array([theta])
    #else:
    #    theta = np.array(theta, ndmin=1)

    #theta = np.atleast_2d(theta).T if theta.ndim == 1 else theta

    TT, NN = data_u.shape

    group_code = np.arange(1, NN + 1)
    Ngroup = max(group_code)

    epsi = 0.001  # Step size of the numerical derivative for score

    alpha, beta, nuinv_z, nuinv_eps, psi_z = theta.flatten()

    omega = (1 - beta) * np.log(lam_bar)

    LL = np.full(TT, np.nan)

    lam_ini = np.clip(lam_ini, 0.01, 4)

    lam = np.full((TT, Ngroup), np.nan)
    lam[0, :] = lam_ini.T

    log_lam = np.full((TT, Ngroup), np.nan)
    log_lam[0, :] = np.log(lam_ini)

    s = np.full((TT, Ngroup), np.nan)

    # Evaluate the marginal cdf (G) and pdf (g) of skew t - t factor model at various x and factor loadings fixing other parameters
    x1 = -15
    x2 = 15
    Npoints = 100
    x_grid = np.linspace(x1, x2, Npoints)
    x_grid = np.concatenate(([-30], x_grid, [30]))

    lam_grid = np.concatenate(([0.001, 0.01], np.arange(0.05, 2.55, 0.05)))

    Gcdf_ini, Gpdf_ini = Gcdfpdf_Skewtt([nuinv_z, nuinv_eps, psi_z],  x_grid, lam_grid) #GLweight, x_grid, lam_grid)

    # Interpolate the marginal cdf and pdf along with finer grids of factor loadings (lam)
    dense_lam_grid = np.concatenate(([0.001], np.arange(0.01, 2.51, 0.001)))
    Gcdf = np.full((len(x_grid), len(dense_lam_grid)), np.nan)
    Gpdf = np.full((len(x_grid), len(dense_lam_grid)), np.nan)

    for i in range(len(dense_lam_grid)):
        interp_func_cdf = interp2d(lam_grid, x_grid, Gcdf_ini, kind='linear')
        interp_func_pdf = interp2d(lam_grid, x_grid, Gpdf_ini, kind='linear')
        Gcdf[:, i] = interp_func_cdf(dense_lam_grid[i], x_grid)
        Gpdf[:, i] = interp_func_pdf(dense_lam_grid[i], x_grid)

    # Evaluate log density of skew t - t factor copula with GAS recursion at each time t
    for tt in range(TT):
        if tt != 0:
            log_lam[tt, :] = omega + alpha * s[tt - 1, :] + beta * log_lam[tt - 1, :]
            lam[tt, :] = np.exp(log_lam[tt, :])

        lam[tt, lam[tt, :] < 0.01] = 0.01
        lam[tt, lam[tt, :] > 2.5] = 2.5

        L_temp, N_derivative = LL_eacht_GASFacCop_Skewtt_Ngroup(
            np.concatenate(([lam[tt, :]], [nuinv_z, nuinv_eps, psi_z])).flatten(),
            data_u[tt, :].flatten(),
            Gcdf,
            Gpdf,
            x_grid,
            dense_lam_grid,
            group_code,
            epsi
        )

        LL[tt] = L_temp
        s[tt, :] = N_derivative.flatten() * lam[tt, :]

    out = -np.sum(LL)

    return out, LL, lam, log_lam, s





def LL_GASFacCop_Skewtt_NGroup(theta, data_u, group_code, lam_ini):
    """
    This function computes the sum of (negative) log likelihoods of factor copula (skew t - t) with GAS recursion.

    Inputs:
        theta - A vector of parameters [omega1, omega2, ..., omegaN, alpha, beta, nuinv_z, nuinv_eps, psi_z]
        data_u - A TxN matrix, the matrix of Unif(0,1) to be modeled with this copula
        
        group_code - A Nx1 (or 1xN) vector of group codes into which each firm is classified
        lam_ini - A (N_group x 1) vector of factor loadings at t=1

    Outputs:
        out - A scalar, the sum of (negative) log-likelihoods of factor copula evaluated at each time t
        LL - A Tx1 vector of log-likelihoods of factor copula evaluated at each time t
        lam - A (T x N_group) matrix of (time-varying) factor loadings
        log_lam - A (T x N_group) matrix of (time-varying) log of factor loadings
        s - A (T x N_group) matrix of (time-varying) score
    """

    TT, NN = data_u.shape
    Ngroup = max(group_code)

    if len(group_code) != NN:
        raise ValueError('The length of the vector "group_code" should equal the dimension of cross sections')

    if Ngroup != (len(theta) - 5):
        raise ValueError('The maximum number of the vector "group_code" should be the same as the number of omegas')
    
    # Ensure theta is a numpy array and a column vector
    #if np.isscalar(theta):
    #    theta = np.array([theta])
    #else:
    #    theta = np.array(theta, ndmin=1)

   # theta = np.atleast_2d(theta).T if theta.ndim == 1 else theta
   

    epsi = 0.001  # Step size of the numerical derivative for score

    omega = theta[:Ngroup]
    alpha = theta[-5]
    beta = theta[-4]
    nuinv_z = theta[-3]
    nuinv_eps = theta[-2]
    psi_z = theta[-1]

    LL = np.full(TT, np.nan)
    lam = np.full((TT, Ngroup), np.nan)
    lam[0, :] = lam_ini

    log_lam = np.full((TT, Ngroup), np.nan)
    log_lam[0, :] = np.log(lam_ini)

    s = np.full((TT, Ngroup), np.nan)

    # Evaluate the marginal cdf (G) and pdf (g) of skew t - t factor model at various x and factor loadings fixing other parameters
    x1 = -15
    x2 = 15
    Npoints = 100
    x_grid = np.linspace(x1, x2, Npoints)
    x_grid = np.concatenate(([-30], x_grid, [30]))

    lam_grid = np.concatenate(([0.001, 0.01], np.arange(0.05, 2.55, 0.05)))

    Gcdf_ini, Gpdf_ini = Gcdfpdf_Skewtt([nuinv_z, nuinv_eps, psi_z],x_grid, lam_grid) #GLweight, x_grid, lam_grid)

    # Interpolate the marginal cdf and pdf along with finer grids of factor loadings (lam)
    dense_lam_grid = np.concatenate(([0.001], np.arange(0.01, 2.51, 0.001)))
    Gcdf = np.full((len(x_grid), len(dense_lam_grid)), np.nan)
    Gpdf = np.full((len(x_grid), len(dense_lam_grid)), np.nan)

    for i in range(len(dense_lam_grid)):
        interp_func_cdf = interp2d(lam_grid, x_grid, Gcdf_ini, kind='linear')
        interp_func_pdf = interp2d(lam_grid, x_grid, Gpdf_ini, kind='linear')
        Gcdf[:, i] = interp_func_cdf(dense_lam_grid[i], x_grid)
        Gpdf[:, i] = interp_func_pdf(dense_lam_grid[i], x_grid)

    # Evaluate log density of skew t - t factor copula with GAS recursion at each time t
    for tt in range(TT):
        if tt != 0:
            log_lam[tt, :] = omega + alpha * s[tt - 1, :] + beta * log_lam[tt - 1, :]
            lam[tt, :] = np.exp(log_lam[tt, :])

        lam[tt, lam[tt, :] < 0.01] = 0.01
        lam[tt, lam[tt, :] > 2.5] = 2.5

        L_temp, N_derivative = LL_eacht_GASFacCop_Skewtt_Ngroup(
            np.concatenate(([lam[tt, :]], [nuinv_z, nuinv_eps, psi_z])).flatten(),
            data_u[tt, :].flatten(),
            Gcdf,
            Gpdf,
            x_grid,
            dense_lam_grid,
            group_code,
            epsi
        )

        LL[tt] = L_temp
        s[tt, :] = N_derivative.flatten() * lam[tt, :]

    out = -np.sum(LL)

    return out, LL, lam, log_lam, s




