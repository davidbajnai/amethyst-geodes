import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib import transforms

def prime(x):
    return 1000 * np.log(x / 1000 + 1)


def unprime(x):
    return (np.exp(x / 1000) - 1) * 1000


def Dp17O(d17O, d18O):
    return (prime(d17O) - 0.528 * prime(d18O)) * 1000


def d17O(d18O, Dp17O):
    return unprime(Dp17O / 1000 + 0.528 * prime(d18O))


def mix_d17O(d18O_A, d17O_A=None, D17O_A=None, d18O_B=None, d17O_B=None, D17O_B=None, step=100):
    ratio_B = np.arange(0, 1+1/step, 1/step)

    if d17O_A is None:
        d17O_A = unprime(D17O_A/1000 + 0.528 * prime(d18O_A))

    if d17O_B is None:
        d17O_B = unprime(D17O_B/1000 + 0.528 * prime(d18O_B))

    mix_d18O = ratio_B * float(d18O_B) + (1 - ratio_B) * float(d18O_A)
    mix_d17O = ratio_B * float(d17O_B) + (1 - ratio_B) * float(d17O_A)
    mix_D17O = Dp17O(mix_d17O, mix_d18O)
    xB = ratio_B * 100

    df = pd.DataFrame(
        {'mix_d17O': mix_d17O, 'mix_d18O': mix_d18O, 'mix_Dp17O': mix_D17O, 'xB': xB})
    return df



def confidence_ellipse(x, y, ax, n_std=2, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns:
    --------
    ell_radius_x, ell_radius_y : float
        Radii of the ellipse before scaling (raw covariance radii).
    scale_x, scale_y : float
        Scaled semi-major and semi-minor axes (after applying standard deviations).
    mean_x, mean_y : float
        Mean of x and y (center of the ellipse).
    transform : matplotlib.transforms.Affine2D
        Full affine transformation applied to the ellipse.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    # Ellipse radii before scaling (raw covariance ellipse)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Create the ellipse
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Standard deviations and scaling
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    # Apply the transformation (rotation, scaling, translation)
    rotation = np.deg2rad(45)  # Rotation in radians
    transf = transforms.Affine2D() \
        .rotate(rotation) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return ell_radius_x, ell_radius_y, scale_x, scale_y, mean_x, mean_y, transf


def apply_theta(d18O_A, Dp17O_A, d18O_B=None, shift_d18O=None, theta=None):

    if d18O_B == None:
        d18O_B = d18O_A + shift_d18O
    d17O_A = d17O(d18O_A, Dp17O_A)

    a18 = (d18O_B + 1000) / (d18O_A + 1000)
    a17 = a18**theta

    d17O_B = a17 * (d17O_A + 1000) - 1000
    Dp17O_B = Dp17O(d17O_B, d18O_B)

    return Dp17O_B
