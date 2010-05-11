"""Routines for fitting beam patterns and baselines."""

import numpy as np
import scipy.special as special
from katpoint import rad2deg

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

def width_string(beamwidth):
    """Pretty-print beamwidth."""
    beamwidth = rad2deg(np.array(beamwidth))
    return ('%.3f deg' % beamwidth) if np.isscalar(beamwidth) else ('(%.3f, %.3f) deg' % (beamwidth[0], beamwidth[1]))

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
    center : sequence of 2 floats
        Initial guess of 2-element beam center, in target coordinate units
    width : sequence of 2 floats, or float
        Initial guess of single beamwidth for both dimensions, or 2-element
        beamwidth vector, expressed as FWHM in units of target coordinates
    height : float
        Initial guess of beam pattern amplitude or height

    Arguments
    ---------
    expected_width : real array, shape (2,), or float
        Initial guess of beamwidth, saved as expected width for checks
    radius_first_null : float
        Radius of first null in beam in target coordinate units (stored here for
        convenience, but not calculated internally)
    refined : int
        Number of scan-based baselines used to refine beam (0 means unrefined)
    is_valid : bool
        True if beam parameters are within reasonable ranges after fit

    """
    def __init__(self, center, width, height):
        ScatterFit.__init__(self)
        if not np.isscalar(width):
            width = np.atleast_1d(width)
        self._interp = GaussianFit(center, fwhm_to_sigma(width) ** 2.0, height)
        self.center = self._interp.mean
        self.width = sigma_to_fwhm(np.sqrt(self._interp.var))
        self.height = self._interp.height

        self.expected_width = width
        # Initial guess for radius of first null
        ##POTENTIAL TWEAK##
        self.radius_first_null = 1.3 * np.mean(self.expected_width)
        # Beam initially unrefined and invalid
        self.refined = 0
        self.is_valid = False

    def fit(self, x, y):
        """Fit a beam pattern to data.

        The center, width and height of the fitted beam pattern can be obtained
        from the corresponding member variables after this is run.

        Parameters
        ----------
        x : array-like, shape (2, N)
            Sequence of 2-dimensional target coordinates (as column vectors)
        y : array-like, shape (N,)
            Sequence of corresponding total power values to fit

        """
        self._interp.fit(x, y)
        self.center = self._interp.mean
        self.width = sigma_to_fwhm(np.sqrt(self._interp.var))
        self.height = self._interp.height
        self.is_valid = not np.any(np.isnan(self.center)) and (self.height > 0.0)
        ##POTENTIAL TWEAK##
        if np.isscalar(self.width):
            self.is_valid = self.is_valid and (self.width > 0.9 * self.expected_width) and \
                                              (self.width < 1.25 * self.expected_width)
        else:
            self.is_valid = self.is_valid and (self.width[0] > 0.9 * self.expected_width[0]) and \
                                              (self.width[0] < 1.25 * self.expected_width[0]) and \
                                              (self.width[1] > 0.9 * self.expected_width[1]) and \
                                              (self.width[1] < 1.25 * self.expected_width[1])

    def __call__(self, x):
        """Evaluate fitted beam pattern function on new target coordinates.

        Parameters
        ----------
        x : array-like, shape (2, M)
            Sequence of 2-dimensional target coordinates (as column vectors)

        Returns
        -------
        y : array, shape (M,)
            Sequence of total power values representing fitted beam

        """
        return self._interp(x)

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
    expected_width : sequence of 2 floats, or float
        Expected beamwidth based on antenna diameter, expressed as FWHM in
        radians. It allows for different widths in the two target coordinates.
        If this is a single number, a circular beam will be fit, while two
        numbers will result in an elliptical beam.
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
    initial_baseline : :class:`fitting.Polynomial2DFit` object
        Initial 2-D polynomial baseline, one for whole compound scan

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
    # Clean up total power
    scan_power = [remove_spikes(scan.stokes(stokes)[:, band]) for scan in compscan.scans]
    total_power = np.hstack(scan_power)
    target_coords = np.hstack([scan.target_coords for scan in compscan.scans])

    # Do initial beam + baseline fitting, where both are fitted in 2-D target coord space
    # This makes no assumptions about the structure of the scans - they are just viewed as a collection of samples
    initial_baseline = Polynomial2DFit(bl_degrees)
    prev_err_power = np.inf
    # Initially, all data is considered to be in the "outer" region and therefore forms part of the baseline
    outer = np.tile(True, len(compscan_power))
    logger.debug("Fitting beam and initial baseline of degree (%d, %d) to pol '%s' of target '%s':" %
                 (bl_degrees[0], bl_degrees[1], underlying_pol, compscan.target.name))
    # Alternate between baseline and beam fitting for a few iterations
    for n in xrange(10):
        initial_baseline.fit(target_coords[:, outer], total_power[outer])
        bl_resid = total_power - initial_baseline(target_coords)
        peak_ind = bl_resid.argmax()
        peak_pos = target_coords[:, peak_ind]
        peak_val = bl_resid[peak_ind]
        beam = BeamPatternFit(peak_pos, expected_width, peak_val)
        beam.fit(target_coords, bl_resid)
        # Calculate Euclidean distance from beam center and identify new "outer" region
        radius = np.sqrt(((target_coords - beam.center[:, np.newaxis]) ** 2).sum(axis=0))
        # This threshold should be close to first nulls of beam - too wide compromises baseline fit
        outer = radius > beam.radius_first_null
        # Check if error remaining after baseline and beam fit has converged, and stop if it has
        resid = bl_resid - beam(target_coords)
        err_power = np.dot(resid, resid)
        logger.debug("Iteration %d: residual = %.2f, beam height = %.3f, width = %s, inner region = %d/%d" %
                     (n, (prev_err_power - err_power) / err_power, beam.height,
                      width_string(beam.width), np.sum(~outer), len(outer)))
        if (err_power == 0.0) or (prev_err_power - err_power) / err_power < 1e-5:
            break
        prev_err_power = err_power + 0.0

    # Find first beam null, by moving outward from beam center in radius range where null is expected
    mean_beamwidth = np.mean(beam.width)
    # The average angular distance moved in target space during a single time sample (== scan speed * dump period)
    scan_distance_per_sample = np.median(np.sqrt((np.diff(target_coords, axis=1) ** 2).sum(axis=0)))
    # Radial half-width of annulus centred on beam null, used both to detect null and to initialise scan baseline fit
    # The annulus should be wide enough to contain 4 time samples (2 on each side of null)
    ##POTENTIAL TWEAK##
    annulus_halfwidth = max(0.2 * mean_beamwidth, 2 * scan_distance_per_sample)
    # Calculate expected number of samples in annulus, both on inside and outside of beam null, based on
    # the expected beam center (== target center) and expected beam null radius
    expected_radius = np.sqrt((target_coords ** 2).sum(axis=0))
    expected_inside_count = ((expected_radius > beam.radius_first_null - annulus_halfwidth) &
                             (expected_radius <= beam.radius_first_null)).sum()
    expected_outside_count = ((expected_radius > beam.radius_first_null) &
                              (expected_radius <= beam.radius_first_null + annulus_halfwidth)).sum()
    # Iterate through potential null radius values, which shifts annulus outwards during search
    ##POTENTIAL TWEAK##
    for null in np.arange(1.2, 1.8, 0.01) * mean_beamwidth:
        inside_null = (radius > null - annulus_halfwidth) & (radius <= null)
        outside_null = (radius > null) & (radius <= null + annulus_halfwidth)
        # Stop if the annulus hits the boundary of scanned region before power starts increasing
        # This is detected by a marked decrease in samples inside the annulus
        ##POTENTIAL TWEAK##
        if (inside_null.sum() < 0.2 * expected_inside_count) or (outside_null.sum() < 0.2 * expected_outside_count):
            break
        # Use median to ignore isolated RFI bumps in some scans
        # Stop when total power starts increasing again as a function of radius
        if np.median(total_power[outside_null]) > np.median(total_power[inside_null]):
            break
    beam.radius_first_null = null

    # If requested, refine beam by redoing baseline fits on per-scan basis, this time fitted against time
    # Also redo the beam fit to the inner region of the beam, where the Gaussian assumption is more correct
    # This assumes that the scans are linear and cross the annular region around the beam null
    good_scan_coords, good_scan_resid, baselines = [], [], []
    if refine_beam:
        # Half-width of inner region of the beam. To select an inner region where beam height > 0.5 * max,
        # the half-width has to be the classical HWHM = 0.5 * FWHM. Slightly enlarge this region to include
        # more scans (which improves fit in direction across scans). Also ensure inner region has at least 5
        # samples across it.
        ##POTENTIAL TWEAK##
        inner_halfwidth = max(0.6 * mean_beamwidth, 2.5 * scan_distance_per_sample)
        for n, scan in enumerate(compscan.scans):
            # Identify regions within annulus close to first beam null within the current scan
            radius = np.sqrt(((scan.target_coords - beam.center[:, np.newaxis]) ** 2).sum(axis=0))
            around_null = np.abs(radius - beam.radius_first_null) < annulus_halfwidth
            padded_selection = np.array([False] + around_null.tolist() + [False])
            borders = np.diff(padded_selection).nonzero()[0] + 1
            # Discard scan if it doesn't contain two separate null regions (with sufficient beam area in between)
            ##POTENTIAL TWEAK##
            if (padded_selection[borders].tolist() != [True, False, True, False]) or \
               (borders[2] - borders[1] < 0.1 * inner_halfwidth / scan_distance_per_sample):
                baselines.append(None)
                continue
            # Calculate standard deviation of samples, based on "ideal total-power radiometer equation"
            mean = scan_power[n].min()
            upper = chi2_conf_interval(dof, mean)[1]
            # Move baseline down as low as possible, taking confidence interval into account
            baseline = Polynomial1DFit(max_degree=1)
            for iteration in range(7):
                baseline.fit(scan.timestamps[around_null], scan_power[n][around_null])
                bl_resid = scan_power[n] - baseline(scan.timestamps)
                ##POTENTIAL TWEAK##
                next_around_null = bl_resid < 1.0 * (upper - mean)
                if not next_around_null.any():
                    break
                else:
                    around_null = next_around_null
            # For non-positive polarisation terms, refit baseline to regions around beam null found in total power
            if pol in ('Q', 'U', 'V'):
                baseline.fit(scan.timestamps[around_null], scan_pol[n][around_null])
            baselines.append(baseline)
            # Identify inner region of beam (close to peak) within scan and add to list if any was found
            inner = radius < inner_halfwidth
            if inner.any():
                good_scan_coords.append(scan.target_coords[:, inner])
                good_scan_resid.append(bl_resid[inner])
        # Need at least 2 good scans, since fitting beam to single scan will introduce large errors in orthogonal dir
        if len(good_scan_resid) > 1:
            # Refit beam to inner region across all good scans
            beam.fit(np.hstack(good_scan_coords), np.hstack(good_scan_resid))
            logger.debug("Refinement: beam height = %.3f, width = %s, first null = %.3f deg, based on %d of %d scans" %
                         (beam.height, width_string(beam.width), rad2deg(beam.radius_first_null),
                          len(good_scan_resid), len(compscan.scans)))
            beam.refined = len(good_scan_resid)

    # Attempt to fit initial beam in non-positive pol term (might be a silly idea)
    if pol in ('Q', 'U', 'V') and not refine_beam:
        bl_resid = compscan_pol - initial_baseline(target_coords)
        beam.fit(target_coords, bl_resid)

    # Do final validation of beam fit
    ##POTENTIAL TWEAK##
    if np.isnan(beam.center).any() or np.isnan(beam.width).any() or np.isnan(beam.height):
        beam = None
    else:
        if np.isscalar(expected_width):
            expected_width = [expected_width, expected_width]
        # If beam center is outside the box scanned out by the telescope, mark it as invalid (good idea to redo scans)
        box_leeway = 0.1 * np.array(expected_width)
        scan_box = [target_coords[0].min() - box_leeway[0], target_coords[0].max() + box_leeway[0],
                    target_coords[1].min() - box_leeway[1], target_coords[1].max() + box_leeway[1]]
        if (beam.center[0] < scan_box[0]) or (beam.center[0] > scan_box[1]) or \
           (beam.center[1] < scan_box[2]) or (beam.center[1] > scan_box[3]):
            beam.is_valid = False
        # If the scan is long enough in one or both coordinates to go out to first beam null on both sides of beam,
        # the beam center should be far enough away from the scan start and end to have the nulls inside the scan,
        # otherwise the baseline won't be accurate
        for scan in compscan.scans:
            scan_limits = scan.target_coords[0].min(), scan.target_coords[0].max()
            if (scan_limits[1] - scan_limits[0] > 2.5 * expected_width[0]) and \
               ((beam.center[0] < scan_limits[0] + expected_width[0]) or \
                (beam.center[0] > scan_limits[1] - expected_width[0])):
                beam.is_valid = False
            scan_limits = scan.target_coords[1].min(), scan.target_coords[1].max()
            if (scan_limits[1] - scan_limits[0] > 2.5 * expected_width[1]) and \
               ((beam.center[1] < scan_limits[0] + expected_width[1]) or \
                (beam.center[1] > scan_limits[1] - expected_width[1])):
                beam.is_valid = False
    return beam, baselines, initial_baseline

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
