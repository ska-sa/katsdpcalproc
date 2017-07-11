#! /usr/bin/env python
#
# Perform a collection of offset pointings on the nearest pointing calibrator.
# Obtain interferometric gain solutions from the pipeline.
# Fit primary beams to the gains and calculate pointing offsets from them.
# Store pointing offsets in telstate.
#
# Ludwig Schwardt
# 2 May 2017
#

import time

import numpy as np
from katcorelib.observe import (standard_script_options, verify_and_connect,
                                collect_targets, start_session, user_logger)
from katpoint import (rad2deg, deg2rad, lightspeed, wrap_angle,
                      RefractionCorrection)
from scape.beam_baseline import BeamPatternFit
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# Group the frequency channels into this many sections to obtain pointing fits
NUM_CHUNKS = 16


class NoTargetsUpError(Exception):
    """No targets are above the horizon at the start of the observation."""


class NoGainsAvailableError(Exception):
    """No gain solutions are available from the cal pipeline."""


def get_pols(telstate):
    """Polarisations associated with calibration products."""
    if 'cal_pol_ordering' not in telstate:
        return []
    polprods = telstate['cal_pol_ordering']
    return [prod[0] for prod in polprods if prod[0] == prod[1]]


def get_cal_inputs(telstate):
    """Input labels associated with calibration products."""
    if 'cal_antlist' not in telstate:
        return []
    ants = telstate['cal_antlist']
    pols = get_pols(telstate)
    return [ant + pol for pol in pols for ant in ants]


def get_bpcal_solution(telstate, st, et):
    """Retrieve bandpass calibration solution from telescope state."""
    inputs = get_cal_inputs(telstate)
    if not inputs or 'cal_product_B' not in telstate:
        return {}
    solutions = telstate.get_range('cal_product_B', st=st, et=et)
    if not solutions:
        return {}
    solution = solutions[-1][0]
    return dict(zip(inputs, solution.reshape((solution.shape[0], -1)).T))


def get_gaincal_solution(telstate, st, et):
    """Retrieve gain calibration solution from telescope state."""
    inputs = get_cal_inputs(telstate)
    if not inputs or 'cal_product_G' not in telstate:
        return {}
    solutions = telstate.get_range('cal_product_G', st=st, et=et)
    if not solutions:
        return {}
    solution = solutions[-1][0]
    return dict(zip(inputs, solution.flat))


def get_offset_gains(session, offsets, offset_end_times, track_duration):
    """Extract gains per pointing offset, per receptor and per frequency chunk.

    Parameters
    ----------
    session : :class:`katcorelib.observe.CaptureSession` object
        The active capture session
    offsets : sequence of *N* pairs of float (i.e. shape (*N*, 2))
        Requested (x, y) pointing offsets relative to target, in degrees
    offset_end_times : sequence of *N* floats
        Unix timestamp at the end of each pointing track
    track_duration : float
        Duration of each pointing track, in seconds

    Returns
    -------
    data_points : dict mapping receptor index to (x, y, freq, gain, weight) seq
        Complex gains per receptor, as multiple records per offset and frequency

    """
    telstate = session.telstate
    pols = get_pols(telstate)
    cal_channel_freqs = telstate.get('cal_channel_freqs', 1.0)
    # XXX Bad hack for now to work around cal centre freq issue
    if not np.isscalar(cal_channel_freqs) and cal_channel_freqs[0] == 0:
        cal_channel_freqs += 856e6
    chunk_freqs = cal_channel_freqs.reshape(NUM_CHUNKS, -1).mean(axis=1)
    data_points = {}
    # Iterate over offset pointings
    for offset, offset_end in zip(offsets, offset_end_times):
        offset_start = offset_end - track_duration
        # Obtain interferometric gains per pointing from the cal pipeline
        bp_gains = get_bpcal_solution(telstate, offset_start, offset_end)
        gains = get_gaincal_solution(telstate, offset_start, offset_end)
        # Iterate over receptors
        for a, ant in enumerate(session.observers):
            pol_gain = np.zeros(NUM_CHUNKS)
            pol_weight = np.zeros(NUM_CHUNKS)
            # Iterate over polarisations (effectively over inputs)
            for pol in pols:
                inp = ant.name + pol
                bp_gain = bp_gains.get(inp)
                gain = gains.get(inp)
                if bp_gain is None or gain is None:
                    continue
                masked = np.ma.masked_invalid(bp_gain * gain)
                chunked = masked.reshape(NUM_CHUNKS, -1)
                abs_chunked = np.abs(chunked)
                abs_chunked_mean = np.ma.mean(abs_chunked, axis=1)
                abs_chunked_std = np.ma.std(abs_chunked, axis=1)
                stats_m = []
                stats_s = []
                for m, s in zip(abs_chunked_mean.filled(np.nan),
                                abs_chunked_std.filled(np.nan)):
                    stats_m.append("%4.2f" % (m,))
                    stats_s.append("%4.2f" % (s,))
                stats_m = ' '.join(stats_m)
                stats_s = ' '.join(stats_s)
                bp_mean = np.nanmean(np.abs(bp_gain))
                user_logger.debug("%s %s %4.2f %s",
                                  tuple(offset), inp, np.abs(gain), stats_m)
                user_logger.debug("%s %s %4.2f %s",
                                  tuple(offset), inp, bp_mean, stats_s)
                avg_gain, weight = np.ma.average(np.abs(chunked),
                                                 axis=1, returned=True)
                # Blend new gains into existing via weighted averaging.
                # XXX We currently combine HH and VV gains at the start to get
                # Stokes I gain but in future it might be better to fit
                # separate beams to HH and VV.
                pol_gain, pol_weight = np.ma.average(
                    np.c_[pol_gain, avg_gain], axis=1,
                    weights=np.c_[pol_weight, weight], returned=True)
            if pol_weight.sum() > 0:
                data = data_points.get(a, [])
                for freq, gain, weight in zip(chunk_freqs, pol_gain,
                                              pol_weight):
                    data.append((offset[0], offset[1],
                                 freq, gain, weight))
                data_points[a] = data
    if not data_points:
        raise NoGainsAvailableError("No gain solutions found in telstate '%s'"
                                    % (session.telstate,))
    return data_points


def fit_primary_beams(session, data_points):
    """Fit primary beams to receptor gains obtained at various offset pointings.

    Parameters
    ----------
    session : :class:`katcorelib.observe.CaptureSession` object
        The active capture session
    data_points : dict mapping receptor index to (x, y, freq, gain, weight) seq
        Complex gains per receptor, as multiple records per offset and frequency

    Returns
    -------
    beams : dict mapping receptor name to list of :class:`scape.beam_baseline.BeamPatternFit`
        Fitted primary beams, per receptor and per frequency chunk

    """
    beams = {}
    # Iterate over receptors
    for a in data_points:
        data = np.rec.fromrecords(data_points[a], names='x,y,freq,gain,weight')
        data = data.reshape(-1, NUM_CHUNKS)
        ant = session.observers[a]
        # Iterate over frequency chunks
        for chunk in range(NUM_CHUNKS):
            chunk_data = data[:, chunk]
            is_valid = np.nonzero(~np.isnan(chunk_data['gain']) &
                                  (chunk_data['weight'] > 0.))[0]
            chunk_data = chunk_data[is_valid]
            if len(chunk_data) == 0:
                continue
            expected_width = rad2deg(ant.beamwidth * lightspeed /
                                     chunk_data['freq'][0] / ant.diameter)
            # Convert power beamwidth to gain / voltage beamwidth
            expected_width = np.sqrt(2.0) * expected_width
            # XXX This assumes we are still using default ant.beamwidth of 1.22
            # and also handles larger effective dish diameter in H direction
            expected_width = (0.8 * expected_width, 0.9 * expected_width)
            beam = BeamPatternFit((0., 0.), expected_width, 1.0)
            x = np.c_[chunk_data['x'], chunk_data['y']].T
            y = chunk_data['gain']
            std_y = np.sqrt(1. / chunk_data['weight'])
            try:
                beam.fit(x, y, std_y)
            except TypeError:
                continue
            beamwidth_norm = beam.width / np.array(expected_width)
            user_logger.debug("%s %2d %2d (%6.2f, %6.2f) %s",
                              ant.name, chunk, len(y), beamwidth_norm[0],
                              beamwidth_norm[1], beam.is_valid)
            # Store data points on beam object to simplify plotting later on
            beam.x = x
            beam.y = y
            beam.std_y = std_y
            # Store beam per frequency chunk and per receptor
            beams_freq = beams.get(ant.name, [None] * NUM_CHUNKS)
            beams_freq[chunk] = beam
            beams[ant.name] = beams_freq
    return beams


def calc_pointing_offsets(session, beams, target, middle_time,
                          temperature, pressure, humidity):
    """Calculate pointing offsets per receptor based on primary beam fits.

    Parameters
    ----------
    session : :class:`katcorelib.observe.CaptureSession` object
        The active capture session
    beams : dict mapping receptor name to list of :class:`scape.beam_baseline.BeamPatternFit`
        Fitted primary beams, per receptor and per frequency chunk
    target : :class:`katpoint.Target` object
        The target on which offset pointings were done
    middle_time : float
        Unix timestamp at the middle of sequence of offset pointings, used to
        find the mean location of a moving target (and reference for weather)
    temperature, pressure, humidity : float
        Atmospheric conditions at middle time, used for refraction correction

    Returns
    -------
    pointing_offsets : dict mapping receptor name to offset data (8 floats)
        Pointing offsets per receptor, stored as a sequence of
          - requested (az, el),
          - full (az, el) adjustment (including pointing model contribution),
          - extra (az, el) adjustment (on top of pointing model), and
          - rough uncertainty of (az, el) adjustment.

    """
    pointing_offsets = {}
    # Iterate over receptors
    for a, ant in enumerate(session.observers):
        beams_freq = beams[ant.name]
        beams_freq = [b for b in beams_freq if b is not None and b.is_valid]
        if not beams_freq:
            user_logger.debug("%s has no valid primary beam fits", ant.name)
            continue
        offsets_freq = np.array([b.center for b in beams_freq])
        offsets_freq_std = np.array([b.std_center for b in beams_freq])
        weights_freq = 1. / offsets_freq_std ** 2
        # Do weighted average of offsets over frequency chunks
        results = np.average(offsets_freq, axis=0, weights=weights_freq,
                             returned=True)
        pointing_offset = results[0]
        pointing_offset_std = np.sqrt(1. / results[1])
        user_logger.debug("%s x=%+7.2f'+-%.2f' y=%+7.2f'+-%.2f'", ant.name,
                          pointing_offset[0] * 60, pointing_offset_std[0] * 60,
                          pointing_offset[1] * 60, pointing_offset_std[1] * 60)
        # Start with requested (az, el) coordinates, as they apply
        # at the middle time for a moving target
        requested_azel = target.azel(timestamp=middle_time, antenna=ant)
        # Correct for refraction, which becomes the requested value
        # at input of pointing model
        rc = RefractionCorrection()
        def refract(az, el):  # noqa: E301, E306
            """Apply refraction correction as at the middle of scan."""
            return [az, rc.apply(el, temperature, pressure, humidity)]
        requested_azel = np.array(refract(*requested_azel))
        pointed_azel = np.array(ant.pointing_model.apply(*requested_azel))
        # Convert fitted offset back to spherical (az, el) coordinates
        pointing_offset = deg2rad(np.array(pointing_offset))
        beam_center_azel = target.plane_to_sphere(*pointing_offset,
                                                  timestamp=middle_time,
                                                  antenna=ant)
        # Now correct the measured (az, el) for refraction and then
        # apply the old pointing model to get a "raw" measured (az, el)
        # at the output of the pointing model
        beam_center_azel = refract(*beam_center_azel)
        beam_center_azel = ant.pointing_model.apply(*beam_center_azel)
        # Make sure the offset is a small angle around 0 degrees
        full_offset_azel = wrap_angle(beam_center_azel - requested_azel)
        extra_offset_azel = wrap_angle(beam_center_azel - pointed_azel)
        # Cheap 'n' cheerful way to convert cross-el uncertainty to azim form
        offset_azel_std = pointing_offset_std / \
            np.array([np.cos(requested_azel[1]), 1.])
        # We store both the "full" offset including pointing model effects
        # (useful for fitting new PMs) and the "extra" offset on top of the
        # existing PM (actual adjustment for reference pointing)
        point_data = np.r_[rad2deg(requested_azel), rad2deg(full_offset_azel),
                           rad2deg(extra_offset_azel), offset_azel_std]
        pointing_offsets[ant.name] = point_data
    return pointing_offsets


def save_pointing_offsets(session, pointing_offsets, middle_time):
    """Save pointing offsets to telstate and display to user.

    Parameters
    ----------
    session : :class:`katcorelib.observe.CaptureSession` object
        The active capture session
    pointing_offsets : dict mapping receptor name to offset data (8 floats)
        Pointing offsets per receptor
    middle_time : float
        Unix timestamp at the middle of sequence of offset pointings

    """
    user_logger.info("Ant, requested (az, el),   full offset incl PM,  "
                     "extra offset on top of PM,  standard dev")
    for ant in session.observers:
        try:
            offsets = pointing_offsets[ant.name].copy()
        except KeyError:
            user_logger.info('%s has no valid primary beam fit',
                             ant.name)
        else:
            sensor_name = '%s_pointing_offsets' % (ant.name,)
            session.telstate.add(sensor_name, offsets, middle_time)
            # Display all offsets in arcminutes
            offsets[2:] *= 60.
            user_logger.info(u"%s (%+6.2f\u00B0, %5.2f\u00B0) -> "
                             "(%+7.2f', %+7.2f')  =  (%+7.2f', %+7.2f') "
                             "+- (%.2f', %.2f')", ant.name, *offsets)


def plot_primary_beam_fits(session, beams, max_extent):
    """Plot primary beam fits per receptors (x and y scan directions).

    Parameters
    ----------
    session : :class:`katcorelib.observe.CaptureSession` object
        The active capture session
    beams : dict mapping receptor name to list of :class:`scape.beam_baseline.BeamPatternFit`
        Fitted primary beams, per receptor and per frequency chunk
    max_extent : float
        Maximum distance of pointings away from target, in degrees

    """
    fine_scan = np.linspace(-1.2 * max_extent, 1.2 * max_extent, 200)
    fine_offsets_along_x = np.c_[fine_scan, np.zeros_like(fine_scan)]
    fine_offsets_along_y = np.c_[np.zeros_like(fine_scan), fine_scan]
    fig, ax = plt.subplots(len(beams), 2, sharex=True, sharey=True)
    ax[-1, 0].set_xlabel('Pointings along x / az (degrees)')
    ax[-1, 1].set_xlabel('Pointings along y / el (degrees)')
    # Iterate over receptors
    for a, ant in enumerate(session.observers):
        beams_freq = beams[ant.name]
        ax_x = ax[a, 0]
        ax_y = ax[a, 1]
        ax_y.set_ylabel(ant.name, rotation='horizontal', ha='right', va='top')
        # Iterate over frequency chunks
        for chunk in range(NUM_CHUNKS):
            beam = beams_freq[chunk]
            if beam is None:
                continue
            np.nonzero(beam.x[0])[0]
            alpha = 1.0 if beam.is_valid else 0.3
            ls = '-' if beam.is_valid else '--'
            marker = 'o' if beam.is_valid else 's'
            fine_x_gains = beam(fine_offsets_along_x.T)
            scale = fine_x_gains.max()
            active_points = np.nonzero(beam.x[0])[0]
            x_dir = slice(active_points[0], active_points[-1] + 1)
            active_points = np.nonzero(beam.x[1])[0]
            y_dir = slice(active_points[0], active_points[-1] + 1)
            ax_x.plot(fine_scan, fine_x_gains / scale,
                      linestyle=ls, alpha=alpha)
            ax_x.plot(beam.x[0, x_dir], beam.y[x_dir] / scale,
                      marker=marker, alpha=alpha)
            ax_x.set_ylim(0, 1.1)
            ax_x.set_yticks([])
            fine_y_gains = beam(fine_offsets_along_y.T)
            scale = fine_y_gains.max()
            ax_y.plot(fine_scan, fine_y_gains / scale,
                      linestyle=ls, alpha=alpha)
            ax_y.plot(beam.x[1, y_dir], beam.y[y_dir] / scale,
                      marker=marker, alpha=alpha)
            ax_y.set_ylim(0, 1.1)
            ax_y.set_yticks([])


# Set up standard script options
usage = "%prog [options] <'target/catalogue'> [<'target/catalogue'> ...]"
description = 'Perform offset pointings on the first source and obtain ' \
              'pointing offsets based on interferometric gains. At least ' \
              'one target must be specified.'
parser = standard_script_options(usage, description)
# Add experiment-specific options
parser.add_option('-t', '--track-duration', type='float', default=20.0,
                  help='Duration of each offset pointing, in seconds (default=%default)')
parser.add_option('--max-extent', type='float', default=1.0,
                  help='Maximum distance of offset from target, in degrees')
parser.add_option('--pointings', type='int', default=10,
                  help='Number of offset pointings')
# Set default value for any option (both standard and experiment-specific options)
parser.set_defaults(description='Reference pointing', nd_params='off')
# Parse the command line
opts, args = parser.parse_args()

if len(args) == 0:
    raise ValueError("Please specify at least one target argument via name "
                     "('Cygnus A'), description ('azel, 20, 30') or catalogue "
                     "file name ('sources.csv')")

# Build up sequence of pointing offsets running linearly in x and y directions
scan = np.linspace(-opts.max_extent, opts.max_extent, opts.pointings // 2)
offsets_along_x = np.c_[scan, np.zeros_like(scan)]
offsets_along_y = np.c_[np.zeros_like(scan), scan]
offsets = np.r_[offsets_along_y, offsets_along_x]
offset_end_times = np.zeros(len(offsets))
middle_time = 0.0
weather = {}

# Check options and build KAT configuration, connecting to proxies and clients
with verify_and_connect(opts) as kat:
    observation_sources = collect_targets(kat, args)
    # Start capture session
    with start_session(kat, **vars(opts)) as session:
        # Quit early if there are no sources to observe
        if len(observation_sources.filter(el_limit_deg=opts.horizon)) == 0:
            raise NoTargetsUpError("No targets are currently visible - "
                                   "please re-run the script later")
        session.standard_setup(**vars(opts))
        session.capture_start()

        # XXX Eventually pick closest source as our target, now take first one
        target = observation_sources.targets[0]
        target.add_tags('bfcal single_accumulation')
        session.label('interferometric_pointing')
        user_logger.info("Initiating interferometric pointing scan on target "
                         "'%s' (%d pointings of %g seconds each)",
                         target.name, len(offsets), opts.track_duration)
        session.track(target, duration=0, announce=False)
        # Point to the requested offsets and collect extra data at middle time
        for n, offset in enumerate(offsets):
            user_logger.info("pointing to offset of (%g, %g) degrees", *offset)
            session.ants.req.offset_fixed(offset[0], offset[1], opts.projection)
            # This track time actually includes the slew to the pointing so the
            # effective duration is less for first pointing in each direction
            time.sleep(opts.track_duration)
            offset_end_times[n] = time.time()
            if n == len(offsets) // 2:
                # Get weather data for refraction correction at middle time
                temperature = kat.sensor.anc_air_temperature.get_value()
                pressure = kat.sensor.anc_air_pressure.get_value()
                humidity = kat.sensor.anc_air_relative_humidity.get_value()
                weather = {'temperature': temperature, 'pressure': pressure,
                           'humidity': humidity}
                middle_time = offset_end_times[n]

        # Perform basic interferometric pointing reduction
        if not kat.dry_run:
            user_logger.info('Retrieving gains, fitting beams, storing offsets')
            data_points = get_offset_gains(session, offsets, offset_end_times,
                                           opts.track_duration)
            beams = fit_primary_beams(session, data_points)
            pointing_offsets = calc_pointing_offsets(session, beams, target,
                                                     middle_time, **weather)
            save_pointing_offsets(session, pointing_offsets, middle_time)
            # if plt:
            #     plot_primary_beam_fits(session, beams, opts.max_extent)
