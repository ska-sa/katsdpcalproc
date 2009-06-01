"""Routines for fitting beam patterns and baselines."""

import numpy as np
import scipy.special as special

from .fitting import ScatterFit, NonLinearLeastSquaresFit
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
#--- CLASS :  BeamBaselineComboFit
#--------------------------------------------------------------------------------------------------

class BeamBaselineComboFit(ScatterFit):
    """Fit Gaussian beam plus polynomial baseline to two-dimensional data.
    
    This fits a two-dimensional Gaussian curve (with diagonal covariance matrix)
    plus a low-order polynomial of form ``f(x) * g(y)`` to (x, y) data. The
    Gaussian bump represents an antenna beam pattern convolved with a point
    source, while the 2-D polynomial represents the baseline power level. The
    underlying optimiser is a modified Levenberg-Marquardt algorithm
    (:func:`scipy.optimize.leastsq`).
    
    Parameters
    ----------
    beam_center : real array-like, shape (2,)
        Initial guess of 2-element Gaussian mean vector
    beam_width : real array-like, shape (2,)
        Initial guess of 2-element variance vector, expressed as FWHM widths
    beam_height : float
        Initial guess of height of Gaussian curve
    poly_x : real array_like, shape (M,)
        Initial guess for polynomial f(x) (highest-order coefficient first)
    poly_y : real array_like, shape (N,)
        Initial guess for polynomial g(y) (highest-order coefficient first)
    
    """
    def __init__(self, beam_center, beam_width, beam_height, poly_x, poly_y):
        ScatterFit.__init__(self)
        def _beam_plus_baseline(p, xy):
            # Calculate 2-dimensional Gaussian curve with diagonal covariance matrix, in vectorised form
            xy_min_mu = xy - p[np.newaxis, :2]
            beam = p[4] * np.exp(-0.5 * np.dot(xy_min_mu * xy_min_mu, p[2:4]))
#            beam = p[4] * jinc(np.sqrt(np.dot(xy_min_mu * xy_min_mu, p[2:4]))) ** 2
            baseline = np.polyval(p[5:5 + len(poly_x)], xy_min_mu[:, 0]) * \
                       np.polyval(p[5 + len(poly_x):],  xy_min_mu[:, 1])
            return beam + baseline
        self.beam_center = np.atleast_1d(np.asarray(beam_center))
        self.beam_width = np.atleast_1d(np.asarray(beam_width))
        self.beam_height = beam_height
        self.poly_x = poly_x
        self.poly_y = poly_y
        # Create parameter vector for optimisation
        params = np.concatenate((self.beam_center, 1.0 / fwhm_to_sigma(self.beam_width) ** 2.0,
                                 [self.beam_height], poly_x, poly_y))
        # Internal non-linear least squares fitter
        self._interp = NonLinearLeastSquaresFit(_beam_plus_baseline, params, method='leastsq')
    
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
        return not np.any(np.isnan(self.beam_center)) and (self.beam_height > 0.0) and \
               (np.max(self.beam_width) < 3.0 * expected_width) and \
               (np.min(self.beam_width) > expected_width / 1.4)
    
    def fit(self, x, y):
        """Fit a polynomial baseline plus Gaussian bump to data.
        
        The mean, variance and height of the Gaussian bump and the polynomial
        baselines can be obtained from the corresponding member variables after
        this is run.
        
        Parameters
        ----------
        x : array, shape (N, 2)
            Sequence of 2-dimensional target coordinates
        y : array, shape (N,)
            Sequence of corresponding total power values to fit
        
        """
        self._interp.fit(x, y)
        # Recreate Gaussian and polynomial parameters
        self.beam_center = self._interp.params[:2]
        self.beam_width = sigma_to_fwhm(np.sqrt(1.0 / self._interp.params[2:4]))
        self.beam_height = self._interp.params[4]
        self.poly_x = self._interp.params[5:5 + len(self.poly_x)]
        self.poly_y = self._interp.params[5 + len(self.poly_x):]
        
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
    
    def baseline(self, x):
        """Evaluate baseline only at given target coordinates.
        
        Parameters
        ----------
        x : array, shape (M, 2)
            Sequence of 2-dimensional target coordinates
        
        Returns
        -------
        y : array, shape (M,)
            Sequence of total power values representing fitted baseline only
        
        """
        xy_min_mu = x - self.beam_center
        return np.polyval(self.poly_x, xy_min_mu[:, 0]) * np.polyval(self.poly_y,  xy_min_mu[:, 1])

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  fit_beam_and_baseline
#--------------------------------------------------------------------------------------------------

def fit_beam_and_baseline(scan, expected_width, band=0):
    """Simultaneously fit beam and baseline to all subscans in a scan.
    
    This fits a beam pattern and baseline to the total power data in all the
    subscans comprising the scan, as a function of the two-dimensional target
    coordinates. Only one frequency band is used. The power data is smoothed
    before fitting to remove spikes.
    
    Parameters
    ----------
    scan : :class:`scan.Scan` object
        Scan data used to fit beam and baseline
    expected_width : float
        Expected beamwidth based on antenna diameter, expressed as FWHM in radians
    band : int, optional
        Frequency band in which to fit beam and baseline
    
    Returns
    -------
    fitted_beam : :class:`BeamBaselineComboFit` object
        Object that describes fitted beam and baseline
    
    Notes
    -----
    The beam part of the fit fits a Gaussian shape to power data, with the peak
    location as initial mean, and an initial standard deviation based on the
    expected beamwidth. It uses all power values in the fitting process, instead
    of only the points within the half-power beamwidth of the peak as suggested
    in [1]_. This seems to be more robust for weak sources, but with more samples
    close to the peak, things might change again.
    
    .. [1] Ronald J. Maddalena, "Reduction and Analysis Techniques," Single-Dish
       Radio Astronomy: Techniques and Applications, ASP Conference Series,
       vol. 278, 2002.
    
    """
    total_power = np.hstack([remove_spikes(ss.stokes('I')[:, band]) for ss in scan.subscans])
    target_coords = np.hstack([ss.target_coords for ss in scan.subscans])
    peak_ind = total_power.argmax()
    peak_val = total_power[peak_ind]
    peak_pos = target_coords[:, peak_ind]
    power_floor = total_power.min()
    interp = BeamBaselineComboFit(peak_pos, [expected_width] * 2, peak_val - power_floor,
                                  [0.0, 1.0], [0.0, power_floor])
    interp.fit(target_coords.transpose(), total_power)
    return interp
