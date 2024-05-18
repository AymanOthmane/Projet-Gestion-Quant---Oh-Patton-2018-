
import numpy as np
from scipy.stats import t
#from scipy.special import gamma
#from scipy import sqrt
from math import gamma, sqrt, pi

def skewtdis_cdf(x, nu, lambda_):
    """
    Calculate the cumulative distribution function (CDF) for Hansen's skewed t-distribution.
    
    Parameters:
    x : array_like
        Data points where the CDF is evaluated. Can be a matrix, vector, or scalar.
    nu : array_like
        Degrees of freedom parameter, can be a matrix, vector, or scalar.
    lambda_ : array_like
        Skewness parameter, can be a matrix, vector, or scalar.

    Returns:
    np.ndarray
        An array of CDF values at each element of x.
    """
    x, nu, lambda_ = np.broadcast_arrays(x, nu, lambda_)
    nu = np.clip(nu, 2.01, np.inf)  # Prevent invalid nu values

    c = gamma((nu + 1) / 2) / (sqrt(pi * (nu - 2)) * gamma(nu / 2))
    a = 4 * lambda_ * c * ((nu - 2) / (nu - 1))
    b = sqrt(1 + 3 * lambda_**2 - a**2)

    y1 = (b * x + a) / (1 - lambda_) * sqrt(nu / (nu-2))
    y2 = (b * x + a) / (1 + lambda_) * sqrt(nu / (nu-2))

    cdf = np.where(x < -a / b, (1 - lambda_) / 2 * t.cdf(y1, nu), (1 - lambda_) / 2 + (1 + lambda_) / 2 * t.cdf(y2, nu))
    return cdf



# def skewtdis_inv(u, nu, lambda_):
#     """
#     Returns the inverse CDF (quantiles) for Hansen's skewed t-distribution at given probabilities.
    
#     Parameters:
#     u : array_like
#         Probabilities at which to evaluate the inverse CDF (quantiles). Should be in the unit interval (0, 1).
#     nu : array_like
#         Degrees of freedom parameter, can be a matrix or scalar.
#     lambda_ : array_like
#         Skewness parameter, can be a matrix or scalar.

#     Returns:
#     np.ndarray
#         An array of quantiles corresponding to each probability in u.
#     """
#     u = np.atleast_1d(u)
#     nu = np.broadcast_to(nu, u.shape)
#     lambda_ = np.broadcast_to(lambda_, u.shape)

#     c = gamma((nu + 1) / 2) / (sqrt(pi * (nu - 2)) * gamma(nu / 2))
#     a = 4 * lambda_ * c * ((nu - 2) / (nu - 1))
#     b = sqrt(1 + 3 * lambda_**2 - a**2)

#     f1 = u < (1 - lambda_) / 2
#     f2 = ~f1

#     inv1 = np.zeros_like(u)
#     inv2 = np.zeros_like(u)
#     if np.any(f1):
#         inv1[f1] = ((1 - lambda_[f1]) / b[f1]) * sqrt((nu[f1] - 2) / nu[f1]) * t.ppf(u[f1] / (1 - lambda_[f1]), nu[f1]) - a[f1] / b[f1]
#     if np.any(f2):
#         inv2[f2] = ((1 + lambda_[f2]) / b[f2]) * sqrt((nu[f2] - 2) / nu[f2]) * t.ppf(0.5 + (1 / (1 + lambda_[f2])) * (u[f2] - (1 - lambda_[f2]) / 2), nu[f2]) - a[f2] / b[f2]

#     inv = np.where(f1, inv1, inv2)
#     return inv


def skewtdis_inv(u, nu, lambda_):
    """
  #  Returns the inverse CDF (quantiles) for Hansen's skewed t-distribution at given probabilities.
    
  #  Parameters:
  #  u : array_like
  #      Probabilities at which to evaluate the inverse CDF (quantiles). Should be in the unit interval (0, 1).
  #  nu : array_like
  #     Degrees of freedom parameter, can be a matrix or scalar.
   # lambda_ : array_like
  #      Skewness parameter, can be a matrix or scalar.

  #  Returns:
  #  np.ndarray
  #      An array of quantiles corresponding to each probability in u.
  """
    
    u = np.atleast_1d(u)
    if np.isscalar(nu):
        nu = np.full_like(u, nu)
    if np.isscalar(lambda_):
        lambda_ = np.full_like(u, lambda_)

    # Calculer c, a, et b comme des scalaires
    c_scalar = gamma((nu + 1) / 2) / (sqrt(pi * (nu - 2)) * gamma(nu / 2))
    a_scalar = 4 * lambda_ * c_scalar * ((nu - 2) / (nu - 1))
    b_scalar = sqrt(1 + 3 * lambda_ ** 2 - a_scalar ** 2)

    # Convertir c, a, et b en tableaux de la même dimension que u s'ils ne sont pas déjà des tableaux
    if not isinstance(c_scalar, np.ndarray):
        c = np.ones_like(u) * c_scalar
    else:
        c = c_scalar

    if not isinstance(a_scalar, np.ndarray):
        a = np.ones_like(u) * a_scalar
    else:
        a = a_scalar

    if not isinstance(b_scalar, np.ndarray):
        b = np.ones_like(u) * b_scalar
    else:
        b = b_scalar

    # Créer des masques booléens f1 et f2

    f1 = u < (1 - lambda_) / 2
    f2 = ~f1

    inv1 = np.zeros_like(u)
    inv2 = np.zeros_like(u)
    if np.any(f1):
        inv1[f1] = ((1 - lambda_[f1]) / b[f1]) * sqrt((nu[f1] - 2) / nu[f1]) * t.ppf(u[f1] / (1 - lambda_[f1]), nu[f1]) - a[f1] / b[f1]
    if np.any(f2):
        inv2[f2] = ((1 + lambda_[f2]) / b[f2]) * sqrt((nu[f2] - 2) / nu[f2]) * t.ppf(0.5 + (1 / (1 + lambda_[f2])) * (u[f2] - (1 - lambda_[f2]) / 2), nu[f2]) - a[f2] / b[f2]

    inv = np.where(f1, inv1, inv2)
    return inv


def skewtdis_pdf(x, nu, lambda_):
    """
    Returns the probability density function (PDF) of Hansen's skewed t-distribution.
    
    Parameters:
    x : array_like
        Data points where the PDF is evaluated, can be a matrix, vector, or scalar.
    nu : array_like
        Degrees of freedom parameter, can be a matrix, vector, or scalar.
    lambda_ : array_like
        Skewness parameter, can be a matrix, vector, or scalar.

    Returns:
    np.ndarray
        An array of PDF values at each element of x.
    """
    x, nu, lambda_ = np.broadcast_arrays(x, nu, lambda_)

    c = gamma((nu + 1) / 2) / (sqrt(pi * (nu - 2)) * gamma(nu / 2))
    a = 4 * lambda_ * c * ((nu - 2) / (nu - 1))
    b = sqrt(1 + 3 * lambda_**2 - a**2)

    pdf1 = b * c * (1 + 1 / (nu - 2) * ((b * x + a) / (1 - lambda_))**2)**(-(nu + 1) / 2)
    pdf2 = b * c * (1 + 1 / (nu - 2) * ((b * x + a) / (1 + lambda_))**2)**(-(nu + 1) / 2)

    # Using logical conditions to apply different PDF formulas based on x values
    pdf = np.where(x < (-a / b), pdf1, pdf2)

    return pdf

