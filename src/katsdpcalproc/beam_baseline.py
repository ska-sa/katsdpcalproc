"""Routines for fitting beam patterns and baselines."""

import numpy as np
import scipy.special as special

from .fitting import ScatterFit, Polynomial2DFit, GaussianFit
from .stats import remove_spikes

def jinc(x):
    """The ``jinc`` function, a circular analogue to the ``sinc`` function.
    
    This calculates the ``jinc`` function, defined as
    
        ``jinc(x) = 2.0 * J_1 (pi * x) / (pi * x),``
    
    where J_1(x) is the Bessel function of the first kind of order 1. It is
    similar to the more well-known sinc function. The function is vectorised.
    
    """
    if np.isscalar(x):
        if x == 0.0:
            return 1.0
        else:
            return 2.0 * special.j1(np.pi * x) / (np.pi * x)
    else:
        y = np.ones(x.shape)
        nonzero = (x != 0.0)
        y[nonzero] = 2.0 * special.j1(np.pi * x[nonzero]) / (np.pi * x[nonzero])
        return y

def fwhm_to_sigma(fwhm):
    """Standard deviation of Gaussian function with specified FWHM beamwidth.
    
    This returns the standard deviation of a Gaussian beam pattern with a
    specified full-width half-maximum (FWHM) beamwidth. This beamwidth is the
    width between the two points left and right of the peak where the Gaussian
    function attains half its maximum value.
    
    """
    # Gaussian function reaches half its peak value at sqrt(2 log 2)*sigma => should equal beamwidth/2
    return fwhm / 2.0 / np.sqrt(2.0 * np.log(2.0))

def sigma_to_fwhm(sigma):
    """FWHM beamwidth of Gaussian function with specified standard deviation.
    
    This returns the full-width half-maximum (FWHM) beamwidth of a Gaussian beam
    pattern with a specified standard deviation. This beamwidth is the width
    between the two points left and right of the peak where the Gaussian
    function attains half its maximum value.
    
    """
    # Gaussian function reaches half its peak value at sqrt(2 log 2)*sigma => should equal beamwidth/2
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    
#--------------------------------------------------------------------------------------------------
#--- CLASS :  BeamPatternFit
#--------------------------------------------------------------------------------------------------

class BeamPatternFit(ScatterFit):
    """Fit analytic beam pattern to total power data defined on 2-D plane.
    
    This fits a two-dimensional Gaussian curve (with diagonal covariance matrix)
    to total power data as a function of 2-D coordinates. The Gaussian bump
    represents an antenna beam pattern convolved with a point source.
    
    Parameters
    ----------
    center : real array-like, shape (2,)
        Initial guess of 2-element Gaussian mean vector
    width : real array-like, shape (2,)
        Initial guess of 2-element variance vector, expressed as FWHM widths
    height : float
        Initial guess of height of Gaussian curve
    
    """
    def __init__(self, center, width, height):
        ScatterFit.__init__(self)
        width = np.atleast_1d(np.asarray(width))
        self._interp = GaussianFit(center, fwhm_to_sigma(width) ** 2.0, height)
        self.center = self._interp.mean
        self.width = sigma_to_fwhm(np.sqrt(self._interp.var))
        self.height = self._interp.height
    
    def fit(self, x, y):
        """Fit a beam pattern to data.
        
        The center, width and height of the fitted beam pattern can be obtained
        from the corresponding member variables after this is run.
        
        Parameters
        ----------
        x : array, shape (N, 2)
            Sequence of 2-dimensional target coordinates
        y : array, shape (N,)
            Sequence of corresponding total power values to fit
        
        """
        self._interp.fit(x, y)
        self.center = self._interp.mean
        self.width = sigma_to_fwhm(np.sqrt(self._interp.var))
        self.height = self._interp.height
        
    def __call__(self, x):
        """Evaluate function ``y = f(x)`` on new data.
        
        Parameters
        ----------
        x : array, shape (M, 2)
            Sequence of 2-dimensional target coordinates
        
        Returns
        -------
        y : array, shape (M,)
            Sequence of total power values representing fitted beam plus baseline
        
        """
        return self._interp(x)
        
    def is_valid(self, expected_width):
        """Check whether beam parameters are valid and within acceptable bounds.
        
        Parameters
        ----------
        expected_width : float
            Expected FWHM beamwidth
        
        Returns
        -------
        is_valid : bool
            True if beam parameters check out OK.
        
        """
        return not np.any(np.isnan(self.center)) and (self.height > 0.0) and \
               (np.max(self.width) < 3.0 * expected_width) and \
               (np.min(self.width) > expected_width / 1.4)
    
#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  fit_beam_and_baseline
#--------------------------------------------------------------------------------------------------

def fit_beam_and_baseline(compscan, expected_width, bl_degrees=(3, 3), refine_beam=False, band=0):
    """Simultaneously fit beam and baseline to all scans in a compound scan.
    
    This fits a beam pattern and baseline to the total power data in all the
    scans comprising the compound scan, as a function of the two-dimensional
    target coordinates. Only one frequency band is used. The power data is
    smoothed to remove spikes before fitting.
    
    Parameters
    ----------
    scan : :class:`scan.Scan` object
        Scan data used to fit beam and baseline
    expected_width : float
        Expected beamwidth based on antenna diameter, expressed as FWHM in radians
    bl_degrees : list of 2 ints, optional
        Degree of baseline polynomial along x and y target coordinates
    refine_beam : bool, optional
        True if final beam fit is only to points within FWHM region around peak
    band : int, optional
        Frequency band in which to fit beam and baseline
    
    Returns
    -------
    beam : :class:`BeamPatternFit` object
        Object that describes fitted beam
    baseline : :class:`fitting.Polynomial2DFit` object
        Object that describes fitted baseline
    
    Notes
    -----
    The beam part of the fit fits a Gaussian shape to power data, with the peak
    location as initial mean, and an initial standard deviation based on the
    expected beamwidth. It initially uses all power values in the fitting
    process, instead of only the points within the half-power beamwidth of the
    peak as suggested in [1]_. This seems to be more robust for weak sources.
    The *refine_beam* option adds a final fitting step according to [1]_.
    
    .. [1] Ronald J. Maddalena, "Reduction and Analysis Techniques," Single-Dish
       Radio Astronomy: Techniques and Applications, ASP Conference Series,
       vol. 278, 2002.
    
    """
    total_power = np.hstack([remove_spikes(scan.stokes('I')[:, band]) for scan in compscan.scans])
    target_coords = np.hstack([scan.target_coords for scan in compscan.scans])
    baseline = Polynomial2DFit(bl_degrees)
    max_iters = 30
    prev_err_power = np.inf
    resid_change = 1e-5
    outer = np.tile(True, len(total_power))
    for n in xrange(max_iters):
        baseline.fit(target_coords[:, outer], total_power[outer])
        bl_resid = total_power - baseline(target_coords)
        peak_ind = bl_resid.argmax()
        peak_pos = target_coords[:, peak_ind]
        peak_val = bl_resid[peak_ind]
        beam = BeamPatternFit(peak_pos, [expected_width] * 2, peak_val)
        beam.fit(target_coords.transpose(), bl_resid)
        resid = bl_resid - beam(target_coords.transpose())
        dist_to_center = np.sqrt(((target_coords - beam.center[:, np.newaxis]) ** 2).sum(axis=0))
        outer = dist_to_center > 1.3 * expected_width
        err_power = np.dot(resid, resid)
        print n, ", res =", (prev_err_power - err_power) / err_power, ", height =", beam.height, ", outer =", outer.sum()
        if (err_power == 0.0) or (prev_err_power - err_power) / err_power < resid_change:
            break
        prev_err_power = err_power + 0.0
    if refine_beam:
        inner = dist_to_center < expected_width / 2.0
        beam.fit(target_coords[:, inner].transpose(), bl_resid[inner])
    return beam, baseline
