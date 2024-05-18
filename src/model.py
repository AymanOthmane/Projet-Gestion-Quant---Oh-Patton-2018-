import numpy as np
from scipy.stats import t
from src.skew_t import *
from src.tools import *
from scipy.integrate import quad
from scipy.interpolate import splrep, splev



def factor_cop_Gcdf_calc1_Skewtt(u, x, theta):
    """
    Helper function for calculation of integral:
    G(x) = Integral of Feps(x - Fzinv(u)) * du over [0, 1]
    
    Inputs:
        u - A Kx1 vector of values of u (used by numerical integration function).
        x - A scalar, the value of x that we will evaluate G at.
        theta = [sig2z, nuinv1, nuinv2, lam] the parameters of Fz and Feps.
    
    Outputs:
        out1 - A Kx1 vector, the value of the argument of the integral at each value of u.
    """
    
    
    sig2z, nuinv1, nuinv2, lam = theta
    
    # Ensure u is a numpy array
    #u = np.array(u, ndmin=1)
    # Ensure u is a column vector
    #u = np.atleast_2d(u).T if u.ndim == 1 else u

    # Ensure u is a numpy array and a column vector
    if np.isscalar(u):
        u = np.array([u])
    else:
        u = np.array(u, ndmin=1)
    u = np.atleast_2d(u).T if u.ndim == 1 else u
    
    out1 = skewtdis_cdf( x-skewtdis_inv(u,1/nuinv1,lam)*sqrt(sig2z), 1/nuinv2, 0 )
    return out1


def factor_cop_Gcdf_calc1_Skewtt_loading(u,x,theta):
    """
    Helper function for calculation of integral:
    G(x) = Integrate[ Feps(x- lam*Fzinv(u))*du, 0,1]
    
    Inputs:
        u - A Kx1 vector of values of u (used by numerical integration function).
        x - A scalar, the value of x that we will evaluate G at.
        theta - A list [lam, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps.
        theta, =[lam, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps
    
    Outputs:
        out1 - A Kx1 vector, the value of the argument of the integral at each value of u.
    """
    
    lam, nuinv_z, nuinv_eps, psi_z = theta
    
    # Ensure u is a numpy array
    #u = np.array(u, ndmin=1)
    # Ensure u is a column vector
    #u = np.atleast_2d(u).T if u.ndim == 1 else u
    # Ensure u is a numpy array and a column vector
    if np.isscalar(u):
        u = np.array([u])
    else:
        u = np.array(u, ndmin=1)
    u = np.atleast_2d(u).T if u.ndim == 1 else u

    out1 = skewtdis_cdf( x - lam * skewtdis_inv(u, 1/nuinv_z, psi_z), 1/nuinv_eps, 0 )

    return out1


def factor_cop_Gpdf_calc1_Skewtt_loading(u,x,theta):
    """
        Helper function for calculation of integral:
        g(x) = Integrate[ pdf_eps(x- lam*Fzinv(u))*du, 0,1]

    INPUTS:
        u, a Kx1 vector of values of u (used by numerical integration function)
       x, a scalar, the value of x that we will evaluate g at
      theta, =[lam, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps

    OUTPUTS: out1, a Kx1 vector, the value of the argument of the integral at each value of u

    """
    lam, nuinv_z, nuinv_eps, psi_z = theta
    
    # Ensure u is a numpy array
    #u = np.array(u, ndmin=1)
    # Ensure u is a column vector
    #u = np.atleast_2d(u).T if u.ndim == 1 else u

    # Ensure u is a numpy array and a column vector
    if np.isscalar(u):
        u = np.array([u])
    else:
        u = np.array(u, ndmin=1)
    u = np.atleast_2d(u).T if u.ndim == 1 else u

    out1 = skewtdis_pdf( x - lam * skewtdis_inv(u, 1/nuinv_z, psi_z), 1/nuinv_eps, 0)

    return out1


def factor_cop_Gpdf_GL_Skewtt(x, theta):
    
    # The marginal density of X_i associated with skew t - t factor model
    #
    # INPUTS:
    #    x - A scalar, the value of x that we will evaluate g at
    #    theta - A list [lam, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps
    #
    # OUTPUTS:
    #    out1 - A scalar, the value of the marginal pdf at x

    result, error = quad(factor_cop_Gpdf_calc1_Skewtt_loading, 1e-5, 1-1e-5, args=(x, theta))
    return result

def factor_cop_Gcdf_GL_Skewtt(x, theta):
    """
    The marginal CDF of X_i associated with skew t - t factor model.

    Inputs:
        x - A scalar, the value of x that we will evaluate G at
        theta - A list [lam, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps
        GLweight - A matrix, nodes and weights for Gauss-Legendre quadrature

    Outputs:
        out1 - A scalar, the value of the marginal CDF at x
    """
    out1, _ = quad(factor_cop_Gcdf_calc1_Skewtt_loading, 1e-5, 1-1e-5, args=(x, theta))
    return out1


def factor_cop_Gpdfvec_Skewtt(q, theta, Gpdf, x_grid, lam_grid):
    """
    The vector of (approximate) marginal densities at the vector q associated with skew t - t factor model.

    Inputs:
        q - A vector, the values that we will (approximately) evaluate marginal pdfs at
        theta - A list [factor loading, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps
        Gpdf - A matrix of the marginal pdfs of skew t-t factor model at x and factor loading (lam)
        x_grid - A vector of x that Gpdf is evaluated at
        lam_grid - A vector of factor loading that Gpdf is evaluated at

    Outputs:
        out - A vector, the marginal pdfs approximated at q
    """
    # Find the closest lambda value in lam_grid to theta[0]
    inx_a = np.argmin(np.abs(lam_grid - theta[0]))
    Gpdf_appro = Gpdf[:, inx_a]

    # Interpolation using spline
    tck = splrep(x_grid, Gpdf_appro, s=0)
    x = splev(q, tck)

    # Switch to numerical integral if spline doesn't work properly
    inx = np.where(x < 0)[0]
    if len(inx) > 0:
        print('spline failed, switch to numerical integral')
        Wa, Wb = np.polynomial.legendre.leggauss(50)
        GLweight = np.column_stack((Wa, Wb))
        for i in inx:
            x[i], _ = quad(factor_cop_Gpdf_calc1_Skewtt_loading, 1e-5, 1 - 1e-5, args=(q[i], theta))
    
    return x


def factor_cop_Gcdfinv_spline(q, theta, Gcdf, x_grid, lam_grid):
    """
    The inverse cdf of x.

    Inputs:
        q - A vector in (0,1), the value of q that we will evaluate Ginv at
        theta - A list [factor loadings, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps
        Gcdf - A [Num_x_grid by Num_lam_grid] matrix of the marginal cdfs of skew t-t factor model at x and factor loading (lam)
        x_grid - A vector of x that Gcdf is evaluated at
        lam_grid - A vector of factor loading that Gcdf is evaluated at

    Outputs:
        x - A vector, the value of the inverse CDF at q
    """
    N = len(q)

    # Find the closest lambda value in lam_grid to theta[0]
    inx_a = np.argmin(np.abs(lam_grid - theta[0]))
    Gcdf_appro = Gcdf[:, inx_a]

    xx = np.column_stack((x_grid, Gcdf_appro))
    _, b = np.unique(xx[:, 1], return_index=True)  # Just in case xx[:, 2] has identical values
    xx = xx[b]

    # Adjust q to be within the bounds of xx[:, 1]
    q = np.clip(q, np.min(xx[:, 1]), np.max(xx[:, 1]))

    # Interpolation using spline
    tck = splrep(xx[:, 1], xx[:, 0], s=0)
    x4 = splev(q, tck)

    x2a = x4

    # Adding a check that the spline quantiles are monotonic in q
    temp = np.column_stack((np.arange(N), q))
    temp = temp[np.argsort(temp[:, 1])]  # Sorting by q
    x2b = np.sort(x2a)  # Sorting the estimated quantiles
    temp = temp[np.argsort(temp[:, 0])]  # Putting things back in original order

    x = np.zeros_like(q)
    x[temp[:, 0].astype(int)] = x2b

    return x


### ligne des code ci-dessous sont à revoir et peuvent être la cause de quelque bug

def Gcdfpdf_Skewtt(theta, x_grid, lam_grid,GLweight=None):
    """
    The marginal cdf and pdf of X_i evaluated at x_grid and lam_grid (skew t - t factor model).

    Inputs:
        theta - A list [nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps without factor loadings
        GLweight - A matrix, nodes and weights for Gauss-Legendre quadrature
        x_grid - A vector of x that Gcdf and Gpdf are evaluated at
        lam_grid - A vector of factor loading that Gcdf and Gpdf are evaluated at

    Outputs:
        Gcdf - A [Num_x_grid by Num_lam_grid] matrix of the marginal cdfs of skew t-t factor model at x and factor loading (lam)
        Gpdf - A [Num_x_grid by Num_lam_grid] matrix of the marginal pdfs of skew t-t factor model at x and factor loading (lam)
    """
    
    # Initialiser les matrices Gcdf et Gpdf avec des valeurs nan
    Gcdf = np.full((len(x_grid), len(lam_grid)), np.nan)
    Gpdf = np.full((len(x_grid), len(lam_grid)), np.nan)

    # Boucles pour remplir les matrices Gcdf et Gpdf
    for ii in range(len(x_grid)):
        for jj in range(len(lam_grid)):
            Gcdf[ii, jj] = factor_cop_Gcdf_GL_Skewtt(x_grid[ii], [lam_grid[jj]] + theta)#, GLweight)
            Gpdf[ii, jj] = factor_cop_Gpdf_GL_Skewtt(x_grid[ii], [lam_grid[jj]] + theta)#, GLweight)

    return Gcdf, Gpdf



def factor_cop_FXpdf_calc1_Skewtt_DiffLoad_VT(u, x, theta, group_code, epsi):
    """
    Calculate the PDF for a skew t - t factor model with different loadings.

    Inputs:
        u - A matrix of values of u (used by numerical integration function)
        x - A matrix, the values of x that we will evaluate the PDF at
        theta - A list of parameters including factor loadings and skew t distribution parameters [lam1, lam2, ..., lamN, nuinv_z, nuinv_eps, psi_z]
        group_code - A vector indicating group membership
        epsi - A small constant for numerical differentiation

    Outputs:
        out - A matrix of calculated PDF values
    """

    Ngroup = max(group_code)
    if len(theta[:-3]) != Ngroup:
        raise ValueError('N_group is not equal to N_theta')

    nuinv_z = theta[-3]
    nuinv_eps = theta[-2]
    psi_z = theta[-1]

    N = x.shape[0]
    Nnodes, k = u.shape
    if k > Nnodes:
        u = u.T
        Nnodes, _ = u.shape

    Fz_inv_u = skewtdis_inv(u, 1 / nuinv_z, psi_z)

    xxx = np.full((Nnodes * N, 1), np.nan)
    for ii in range(N):
        inx = group_code[ii]
        xxx[Nnodes * ii:Nnodes * (ii + 1), 0] = x[ii, 0] - np.sqrt(theta[inx]) * Fz_inv_u

    xxx = np.tile(xxx, (1, Ngroup + 1))

    for ii in range(N):
        inx = group_code[ii]
        xxx[Nnodes * ii:Nnodes * (ii + 1), 1 + inx] = x[ii, 1] - np.sqrt(theta[inx] + epsi) * Fz_inv_u

    xxx = xxx.flatten()

    out_temp = skewtdis_pdf(xxx, 1 / nuinv_eps, 0)
    out_temp = out_temp.reshape(Nnodes * N, Ngroup + 1)

    out = np.full((Nnodes, Ngroup + 1), np.nan)
    for ii in range(Ngroup + 1):
        temp = out_temp[:, ii].reshape(Nnodes, N)
        out[:, ii] = np.prod(temp, axis=1)

    return out


def factor_cop_FXpdf_GL_Skewtt(x, theta, group_code, epsi, GLweight = None):
    """
    The joint density of [X1,...,XN] associated with skew t - t factor model (1st element of out1)
    with other joint densities evaluated at (factor loading + [0,..,eps_i,..,0], nuinv_z, nuinv_eps, psi_z).

    Inputs:
        x - A [N by 2] matrix, the value of x that we will evaluate g(x1,...,xN) at
            1st column: Ginv(u) evaluated at theta (= [factor loading, nuinv_z, nuinv_eps, psi_z])
            2nd column: Ginv(u) evaluated at theta (= [(factor loading + [0,..,eps_i,..,0]), nuinv_z, nuinv_eps, psi_z])
        theta - A list [factor loading, nuinv_z, nuinv_eps, psi_z], the parameters of Fz and Feps
        GLweight - A matrix, nodes and weights for Gauss-Legendre quadrature
        group_code - A Nx1 vector of group codes into which each firm is classified
        epsi - A scalar, the step size for numerical derivative for score calculation

    Outputs:
        out1 - A [(1 + # of group) by 1] vector;
            1st element: the value of the joint pdf of skew t-t evaluated at
                theta (= [factor loading, nuinv_z, nuinv_eps, psi_z])
            (i+1)th elements: the value of the joint pdf of skew t-t evaluated at
                theta (= [factor loading + [0,..,eps_i,..,0], nuinv_z, nuinv_eps, psi_z])
    """
    
    #def integrand(u, x, theta, group_code, epsi):
    #    return factor_cop_FXpdf_calc1_Skewtt_DiffLoad_VT(u, x, theta, group_code, epsi)

    result, _ = quad(factor_cop_FXpdf_calc1_Skewtt_DiffLoad_VT, 1e-5, 1-1e-5, args=(x, theta, group_code, epsi))
    return result

















