"""Routines for fitting beam patterns and baselines."""

import numpy as np
import scipy.special as special

from .fitting import ScatterFit, Polynomial1DFit, Polynomial2DFit, GaussianFit, Delaunay2DScatterFit
from .stats import remove_spikes, chi2_conf_interval

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
        Initial guess of 2-element beam center, in target coordinate units
    width : real array-like, shape (2,), or float
        Initial guess of single beamwidth for both dimensions, or 2-element
        beamwidth vector, expressed as FWHM in units of target coordinates
    height : float
        Initial guess of beam pattern amplitude or height
    
    Arguments
    ---------
    expected_width : float
        Initial guess of beamwidth, saved as expected average width for checks
    radius_first_null : float
        Radius of first null in beam in target coordinate units (stored here for
        convenience, but not calculated internally)
    
    """
    def __init__(self, center, width, height):
        ScatterFit.__init__(self)
        if not np.isscalar(width):
            width = np.atleast_1d(np.asarray(width))
        self._interp = GaussianFit(center, fwhm_to_sigma(width) ** 2.0, height)
        self.center = self._interp.mean
        self.width = sigma_to_fwhm(np.sqrt(self._interp.var))
        self.height = self._interp.height
        
        self.expected_width = np.mean(width)
        # Initial guess for radius of first null
        self.radius_first_null = 1.3 * self.expected_width
    
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
        
    def is_valid(self):
        """Check whether beam parameters are valid and within acceptable bounds."""
        return not np.any(np.isnan(self.center)) and (self.height > 0.0) and \
               (np.min(self.width) > 0.9 * self.expected_width) and \
               (np.max(self.width) < 1.25 * self.expected_width) 

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  fit_beam_and_baselines
#--------------------------------------------------------------------------------------------------

def fit_beam_and_baselines(compscan, expected_width, dof, band=0):
    """Simultaneously fit beam and baselines to all scans in a compound scan.
    
    This fits a beam pattern and baselines to the total power data in all the
    scans comprising the compound scan. The beam pattern is a Gaussian function
    of the two-dimensional target coordinates for the entire compound scan,
    while the baselines are first-order polynomial functions of time per scan.
    Only one frequency band is used. The power data is smoothed to remove spikes
    before fitting.
    
    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan data used to fit beam and baselines
    expected_width : float
        Expected beamwidth based on antenna diameter, expressed as FWHM in radians
    dof : float
        Degrees of freedom of chi^2 distribution of total power samples
    band : int, optional
        Frequency band in which to fit beam and baselines
    
    Returns
    -------
    beam : :class:`BeamPatternFit` object
        Object that describes beam fitted across compound scan
    baselines : list of :class:`fitting.Polynomial1DFit` objects
        List of fitted baseline objects, one per scan (or None for scans that
        were not fitted)
    
    Notes
    -----
    The beam part of the fit fits a Gaussian shape to power data, with the peak
    location as initial mean, and an initial standard deviation based on the
    expected beamwidth. It initially uses all power values in the fitting
    process, instead of only the points within the half-power beamwidth of the
    peak as suggested in [1]_. This seems to be more robust for weak sources.
    It also fits a 2-D polynomial as initial baseline, across the entire
    compound scan, as part of an iterative fitting process.
    
    The second stage of the fitting uses the first nulls of the fitted beam to
    fit first-order polynomial baselines per scan, as functions of time. Scans
    that do not contain beam nulls are ignored. The beam is finally refined by
    fitting it only to the inner region of the beam, as in [1]_.
    
    .. [1] Ronald J. Maddalena, "Reduction and Analysis Techniques," Single-Dish
       Radio Astronomy: Techniques and Applications, ASP Conference Series,
       vol. 278, 2002.
    
    """
    scan_power = [remove_spikes(scan.stokes('I')[:, band]) for scan in compscan.scans]
    total_power = np.hstack(scan_power)
    target_coords = np.hstack([scan.target_coords for scan in compscan.scans])
    bl_degrees = (3, 3)
    initial_baseline = Polynomial2DFit(bl_degrees)
    prev_err_power = np.inf
    outer = np.tile(True, len(total_power))
    logger.debug('Fitting beam and initial baseline of degree (%d, %d):' % bl_degrees)
    for n in xrange(10):
        initial_baseline.fit(target_coords[:, outer], total_power[outer])
        bl_resid = total_power - initial_baseline(target_coords)
        peak_ind = bl_resid.argmax()
        peak_pos = target_coords[:, peak_ind]
        peak_val = bl_resid[peak_ind]
        beam = BeamPatternFit(peak_pos, expected_width, peak_val)
        beam.fit(target_coords.transpose(), bl_resid)
        resid = bl_resid - beam(target_coords.transpose())
        radius = np.sqrt(((target_coords - beam.center[:, np.newaxis]) ** 2).sum(axis=0))
        # This threshold should be close to first nulls of beam - too wide compromises baseline fit
        outer = radius > beam.radius_first_null
        err_power = np.dot(resid, resid)
        logger.debug("Iteration %d: residual = %f, beam height = %f, width = %f" %
                     (n, (prev_err_power - err_power) / err_power, beam.height, beam.width))
        if (err_power == 0.0) or (prev_err_power - err_power) / err_power < 1e-5:
            break
        prev_err_power = err_power + 0.0
    
    # Find first null, by moving outward from beam center in radius range where null is expected
    for null in np.arange(1.2, 1.8, 0.01) * beam.width:
        inside_null = (radius > null - 0.2 * beam.width) & (radius <= null)
        outside_null = (radius > null) & (radius <= null + 0.2 * beam.width)
        # Stop if end of scanned region is reached
        if (inside_null.sum() < 20) or (outside_null.sum() < 20):
            break
        # Use median to ignore isolated RFI bumps in some scans
        # Stop when total power starts increasing again as a function of radius
        if np.median(total_power[outside_null]) > np.median(total_power[inside_null]):
            break
    beam.radius_first_null = null
    
    good_scan_coords, good_scan_resid, baselines = [], [], []
    for n, scan in enumerate(compscan.scans):
        radius = np.sqrt(((scan.target_coords - beam.center[:, np.newaxis]) ** 2).sum(axis=0))
        around_null = np.abs(radius - beam.radius_first_null) < 0.2 * beam.width
        padded_selection = np.array([False] + around_null.tolist() + [False])
        borders = np.diff(padded_selection).nonzero()[0] + 1
        if (padded_selection[borders].tolist() != [True, False, True, False]) or (borders[2] - borders[1] < 10):
            baselines.append(None)
            continue
        # Calculate standard deviation of samples, based on "ideal total-power radiometer equation"
        mean = scan_power[n].min()
        lower, upper = chi2_conf_interval(dof, mean)
        # Move baseline down as low as possible, taking confidence interval into account
        baseline = Polynomial1DFit(max_degree=1)
        for iteration in range(7):
            baseline.fit(scan.timestamps[around_null], scan_power[n][around_null])
            bl_resid = scan_power[n] - baseline(scan.timestamps)
            around_null = bl_resid < 1.2 * (upper - mean)
        baselines.append(baseline)
        inner = radius < 0.6 * beam.width
        if inner.any():
            good_scan_coords.append(scan.target_coords[:, inner])
            good_scan_resid.append(bl_resid[inner])
    # Beam height is underestimated, as remove_spikes() flattens beam top - adjust it based on Gaussian beam
    beam.fit(np.hstack(good_scan_coords).transpose(), 1.0047 * np.hstack(good_scan_resid))
    logger.debug("Beam refinement: beam height = %f, width = %f, first null = %f, based on %d of %d scans" % \
                 (beam.height, beam.width, beam.radius_first_null, len(good_scan_resid), len(compscan.scans)))
    
    return beam, baselines

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  interpolate_measured_beam
#--------------------------------------------------------------------------------------------------

def interpolate_measured_beam(x, y, z, num_grid_rows=201):
    """Interpolate measured beam pattern contained in a raster scan.
    
    Parameters
    ----------
    x : array-like of float, shape (N,)
        Sequence of *x* target coordinates
    y : array-like of float, shape (N,)
        Sequence of *y* target coordinates
    z : array-like of float, shape (N,)
        Sequence of *z* measurements
    num_grid_rows : int, optional
        Number of grid points on each axis, referred to as M below
    
    Returns
    -------
    grid_x : array of float, shape (M,)
        Array of *x* coordinates of grid
    grid_y : array of float, shape (M,)
        Array of *y* coordinates of grid
    smooth_z : array of float, shape (M, M)
        Interpolated *z* values, as a matrix
    
    """
    # Set up grid points that include the origin at the center and stays within convex hull of samples
    x_lims = [np.min(x), np.max(x)]
    y_lims = [np.min(y), np.max(y)]
    assert (np.prod(x_lims) < 0.0) and (np.prod(y_lims) < 0.0), 'Raster scans should cross target'
    grid_x = np.abs(x_lims).min() * np.linspace(-1.0, 1.0, num_grid_rows)
    grid_y = np.abs(y_lims).min() * np.linspace(-1.0, 1.0, num_grid_rows)
    
    # Obtain smooth interpolator for z data
    interp = Delaunay2DScatterFit(default_val=0.0, jitter=True)
    interp.fit([x, y], z)
    
    # Evaluate interpolator on grid
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    mesh = np.vstack((mesh_x.ravel(), mesh_y.ravel()))
    smooth_z = interp(mesh).reshape(grid_y.size, grid_x.size)
    
    return grid_x, grid_y, smooth_z
