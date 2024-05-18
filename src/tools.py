import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre

def GLNodeWt(n):
    """
    GLNodeWt - Nodes and weights for Gauss-Legendre quadrature of arbitrary order
               obtained by solving an eigenvalue problem.

    Synopsis:  x, w = GLNodeWt(n)

    Input:     n = order of quadrature rule

    Output:    x = vector of nodes
               w = vector of weights

    Algorithm based on ideas from Golub and Welsch, and Gautschi. For a
    condensed presentation see H.R. Schwarz, "Numerical Analysis: A
    Comprehensive Introduction," 1989, Wiley. Original MATLAB
    implementation by H.W. Wilson and L.H. Turcotte, "Advanced Mathematics
    and Mechanics Applications Using MATLAB," 2nd ed., 1998, CRC Press
    """
    # Calculate the off-diagonal elements of the Jacobi matrix
    #beta = (1.0 / np.arange(1, n)) / np.sqrt(4 * (np.arange(1, n))**2 - 1)
    beta = (1 / np.sqrt(np.abs(4 * (1 - np.arange(1, n)) ** 2 - 1)))
    # Create the symmetric tridiagonal Jacobi matrix
    J = np.diag(beta, -1) + np.diag(beta, 1)
    # Solve the eigenvalue problem
    D, V = np.linalg.eig(J)
    # Sort eigenvalues (nodes) and eigenvectors (weights calculation)
    ix = np.argsort(D)
    x = D[ix]
    w = 2 * (V[0, ix]**2)

    return x, w


def GLquad(f, a=-1, b=1, n=10, *args):
    """
    Perform Gauss-Legendre quadrature on a function 'f' over the interval [a, b].

    Inputs:
        f: a callable, the function to integrate, which must take a vector of values.
        a: float, the lower bound of the interval.
        b: float, the upper bound of the interval.
        n: int, the number of nodes to use in the quadrature.
        args: additional arguments to pass to the function 'f'.

    Output:
        out1: float, the estimated integral of the function 'f' over the interval [a, b].
    """
    x, w = GLNodeWt(n)
    # Transform nodes from [-1, 1] to [a, b]
    x_mapped = (x + 1) * (b - a) / 2 + a
    # Evaluate the function at the transformed nodes
    f_values = f(x_mapped, *args)
    # Compute the integral using the weights and the function values
    integral = np.dot(f_values, w) * (b - a) / 2

    return integral


def nines(*args):
    """
    Returns a matrix or array filled with -999.99.

    Parameters:
        *args: Variable length argument list. Can be a sequence of dimensions or an array-like to specify the shape.
    
    Returns:
        np.ndarray: An array of -999.99s with the specified shape.
    """

    # Handle the case where the first argument is array-like or scalar
    if len(args) == 1:
        if np.isscalar(args[0]):
            # Single scalar, assume a square matrix of that size
            return -999.99 * np.ones((args[0], args[0]))
        else:
            # Array-like, create an array of the same shape
            return -999.99 * np.ones(np.shape(args[0]))
    
    # Handle the case with two or more dimensions specified
    elif len(args) > 1:
        shape = tuple(arg if np.isscalar(arg) else len(arg) for arg in args)
        return -999.99 * np.ones(shape)
    
    # Default case if no arguments are provided
    else:
        return np.array([-999.99])
    



def rhobar2betabar(rhobar):
    N = rhobar.shape[0]
    if N < 3:
        raise ValueError("This mapping requires there to be at least 3 variables.")
    
    theta0 = np.ones((N, 1))

    def rho2theta(rho):
        k = rho.shape[1]
        out1 = -999.99 * np.ones((k * (k - 1) // 2, 1))
        #out1 = nines(np.ones((k * (k - 1) // 2, 1)))
        
        counter = 0
        for ii in range(k):
            for jj in range(ii + 1, k):
                out1[counter] = rho[ii, jj]
                counter += 1
        return out1

    def rhobar2betabar_calc(beta, rhobar):
        Nb = len(beta)
        rho = np.full((Nb, Nb), np.nan)
        for ii in range(Nb):
            for jj in range(ii + 1, Nb):
                rho[ii, jj] = beta[ii] * beta[jj] / np.sqrt((1 + beta[ii]**2) * (1 + beta[jj]**2))
                rho[jj, ii] = rho[ii, jj]
        return np.sum((rho2theta(rho) - rho2theta(rhobar))**2)
    
    # Flatten theta0 to make it 1D as required by `minimize`
    theta0_flat = theta0.flatten()

    # Optimization settings
    options = {'disp': False, 'gtol': 1e-6}
    result = minimize(lambda beta: rhobar2betabar_calc(beta, rhobar), theta0_flat, method='BFGS', options=options)
    if not result.success:
        raise RuntimeError("Optimization did not converge: " + result.message)
    
    return result.x




def fminsearchbnd(fun, x0, LB=None, UB=None, options=None, *args):
    """
    Version améliorée de fminsearchbnd3 pour gérer les types d'entrée correctement.
    """
    # Conversion des entrées en tableaux numpy avec gestion des None
    x0 = np.asarray(x0)
    n = len(x0)
    if LB is None:
        LB = -np.inf * np.ones(n)
    else:
        LB = np.array(LB, dtype=float)
        LB[np.isnan(LB)] = -np.inf  # Gérer les None dans LB
    
    if UB is None:
        UB = np.inf * np.ones(n)
    else:
        UB = np.array(UB, dtype=float)
        UB[np.isnan(UB)] = np.inf  # Gérer les None dans UB

    bounds = [(l, u) for l, u in zip(LB, UB)]

    # Définir une fonction de transformation pour les contraintes
    def bound_transform(x, *args):
        tx = np.copy(x)
        for i in range(len(x)):
            if np.isfinite(LB[i]) and np.isfinite(UB[i]):
                tx[i] = LB[i] + (np.sin(x[i]) + 1) * (UB[i] - LB[i]) / 2
            elif np.isfinite(LB[i]):
                tx[i] = LB[i] + np.exp(x[i])
            elif np.isfinite(UB[i]):
                tx[i] = UB[i] - np.exp(-x[i])
        return fun(tx, *args)

    # Configurer les options par défaut
    if options is None:
        options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-9}

    # Exécuter l'optimisation
    result = minimize(bound_transform, x0, args=args, method='L-BFGS-B', bounds=bounds, options=options)

    # Préparation des résultats
    x = result.x
    fval = result.fun
    exitflag = result.status
    output = {
        'iterations': result.nit,
        'funcount': result.nfev,
        'algorithm': result.success,
        'message': result.message
    }
    return x, fval, exitflag, output


