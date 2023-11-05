#! /usr/bin/env python
#
# Updated refrence pointing script
# This script obtain offsets of a reference pointing observation using it's
# associated token (from the archive)
# The following low-level functions are used to calculate the offsets:
# Extract gains per pointing offset, per receptor and per frequency chunk.
# Fit primary beams to the gains 
# Calculate offsets from fitted beams
# Ludwig Schwardt & Tasmiyah Rawoot
#



import katdal
import katsdpcal
import katpoint
import numpy as np
from katsdpcal.plotting import *
from katsdpcal.report import utc_tstr
import matplotlib.pyplot as plt

from katpoint import (rad2deg, deg2rad, lightspeed, wrap_angle, RefractionCorrection)
from scikits.fitting import ScatterFit, GaussianFit

from katsdpcal.calprocs_dask import wavg
import itertools
import math
get_ipython().magic('matplotlib inline')



"""Retrieve and store necessary parameter values from token

    Parameters
    ----------
    token : String
        Token associated with the observation from archive
        e.g. https://archive-gw-1.kat.ac.za/1699209204/1699209204_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1Q...
     
    NUM_CHUNKS : float
        Group the frequency channels into this many sections to obtain pointing fits
    Returns
    -------
    my_params : Dictionary containing telstate <class 'katdal.sensordata.TelstateToStr'>, 
        antennas <class 'katpoint.Antenna>, target <class 'katpoint.Target> , pointing times, track duration, 
        offset co-ordinates, channel frequencies, pols

    """

def get_parameters_from_token(token,NUM_CHUNKS):
    params={}
    f4=katdal.open(str(token))
    
    f4.select(scans='track')
    params['ants']=f4.ants
    telstate=f4.source.telstate
    params['telstate']=telstate
    params['target']=f4.catalogue.targets[f4.target_indices[0]]
    
    ###Get number of dumps per offset ignoring the final pointing at the target
    no_of_dumps=f4.dumps
    groups = []
    for k, g in itertools.groupby(enumerate(no_of_dumps), lambda x: x[0]-x[1]):
        groups.append(list(map(lambda x: x[1], g)))
    res = [(group[0], group[-1]) for group in groups]
    for i in range(0,len(res)):
        res[i]=(res[i][-1]-res[i][0] +1)
    res=res[0:-1]
    
    ##converting pointing per dump for each 3D array to a 1D array for pointings per track
    x_per_dump=np.mean(f4.target_x,axis=1)

    ##creating indices where to split array of pointings per dump 
    indices=[sum(res[:i+1]) for i in range(len(res))]
    
    q=np.split(x_per_dump,indices)

    x_per_track=np.zeros(len(q),dtype=np.ndarray)
    for i in range(0,len(q)):
        x_per_track[i]=q[i]
    for i in range(0,len(x_per_track)):
        x_per_track[i] =np.median(x_per_track[i])

    y_per_dump=np.mean(f4.target_y,axis=1)
    q=np.split(y_per_dump,indices)
    y_per_track=np.zeros(len(q),dtype=np.ndarray)
    for i in range(0,len(q)):
        y_per_track[i]=q[i]
    for i in range(0,len(y_per_track)):
        y_per_track[i] =np.median(y_per_track[i])
    ###Getting offset co-ordinates (x,y)
    offsets= list(zip(x_per_track, y_per_track))

    offsets=offsets[0:-1]
    params['offsets']=offsets
    
    ###Getting middle and end times for pointing offsets and track duration
    mid_times=[t for (v,t) in f4.source.telstate.get_range("cal_product_B0",st=0)][0:-1]
    params['mid_times']=mid_times
    end_times=[]
    for i in range(0,len(mid_times)):
        end_times.append(math.floor(mid_times[i]+(f4.dump_period*(res[i]-1)*0.5)))
    track_duration=f4.dump_period*(np.array(res)-1)
    params['end_times']=end_times
    params['track_duration']=track_duration
    ###Getting frequencies for chunks
    from katdal.spectral_window import SpectralWindow

    cal_channel_freqs=telstate['cal_center_freq'] +(np.arange(telstate['cal_n_chans']) - telstate['cal_n_chans']/2)*(telstate['cal_bandwidth']/telstate['n_chans'])
    params['cal_channel_freqs']=cal_channel_freqs
    chunk_freqs = cal_channel_freqs.reshape(NUM_CHUNKS, -1).mean(axis=1)
    params['chunk_freqs']=chunk_freqs
    pols =telstate['cal_pol_ordering']
    params['pols']=pols
    
    return params



"""Extract gains per pointing offset, per receptor and per frequency chunk.

    Parameters
    ----------
    my_params : Dictionary containing telstate <class 'katdal.sensordata.TelstateToStr'>, 
        antennas, target, pointing times, track duration etc.
    
    NUM_CHUNKS : float
        Group the frequency channels into this many sections to obtain pointing fits
    Returns
    -------
    data_points : dict mapping receptor index to (x, y, freq, gain, weight) seq
        Complex gains per receptor, as multiple records per offset and frequency

    """

def get_offset_gains(my_params,NUM_CHUNKS):

    data_points = {}   
    for i, j, k in zip(my_params["offsets"],my_params["end_times"],my_params["track_duration"]):
            offset_start = j - k

            # Obtain interferometric gains per pointing from telstate
            try: 
                bp_gains = my_params["telstate"].get_range('cal_product_B0',st=offset_start,et=j)[0][0]
                gains = my_params["telstate"].get_range('cal_product_G',st=offset_start,et=j)[0][0]


            except CalSolutionsUnavailable as err:
                user_logger.warning('Skipping offset %s: %s', offset, err)
                continue
            # Iterate over receptors
            for a, ant in enumerate(my_params["telstate"].cal_antlist):
                        
                        pol_gain = np.zeros(NUM_CHUNKS)
                        pol_weight = np.zeros(NUM_CHUNKS)
                        # Iterate over polarisations (effectively over inputs)
                        for pol in range(0,2):
                            bp_gain = bp_gains[:, pol,a]
                            gain = gains[pol,a]

                            if bp_gain is None or gain is None:
                                continue

                            masked_gain = np.ma.masked_invalid(bp_gain * gain)
                            abs_gain_chunked = np.abs(masked_gain).reshape(NUM_CHUNKS, -1)
                            abs_gain_mean = abs_gain_chunked.mean(axis=1)
                            abs_gain_std = abs_gain_chunked.std(axis=1)
                            abs_gain_var = abs_gain_std.filled(np.inf) ** 2
                            zero_var = abs_gain_var == 0.
                            if all(zero_var):
                                abs_gain_var = np.ones_like(abs_gain_var) * np.inf
                            else:
                                abs_gain_var[zero_var] = abs_gain_var[~zero_var].min()
                            # Number of valid samples going into statistics
                            abs_gain_N = (~abs_gain_chunked.mask).sum(axis=1)
                            # Generate standard precision weights based on empirical stdev
                            abs_gain_weight = abs_gain_N / abs_gain_var
                            # Prepare some debugging output
                            stats_mean = ' '.join("%4.2f" % (m,) for m in
                                                  abs_gain_mean.filled(np.nan))
                            stats_std = ' '.join("%4.2f" % (s,) for s in
                                                 abs_gain_std.filled(np.nan))
                            stats_N = ' '.join("%4d" % (n,) for n in abs_gain_N)
                            bp_mean = np.nanmean(np.abs(bp_gain))
                            # Blend new gains into existing via weighted averaging.
                            # XXX We currently combine HH and VV gains at the start to get
                            # Stokes I gain but in future it might be better to fit
                            # separate beams to HH and VV.
                            pol_gain, pol_weight = np.ma.average(
                                np.c_[pol_gain, abs_gain_mean], axis=1,
                                weights=np.c_[pol_weight, abs_gain_weight], returned=True)
                        if pol_weight.sum() > 0:
                            # Turn masked values into NaNs pre-emptively to avoid warning
                            # when recarray in beam fitting routine forces this later on.
                            pol_gain = pol_gain.filled(np.nan)
                            data = data_points.get(a, [])
                            for freq, gain, weight in zip(my_params["chunk_freqs"], pol_gain, pol_weight):
                                data.append((i[0], i[1], freq, gain, weight))
                            data_points[a] = data
    return data_points



def fwhm_to_sigma(fwhm):
    """Standard deviation of Gaussian function with specified FWHM beamwidth.

    This returns the standard deviation of a Gaussian beam pattern with a
    specified full-width half-maximum (FWHM) beamwidth. This beamwidth is the
    width between the two points left and right of the peak where the Gaussian
    function attains half its maximum value.

    """
    # Gaussian function reaches half its peak value at sqrt(2 log 2)*sigma
    return fwhm / 2.0 / np.sqrt(2.0 * np.log(2.0))

def sigma_to_fwhm(sigma):
    """FWHM beamwidth of Gaussian function with specified standard deviation.

    This returns the full-width half-maximum (FWHM) beamwidth of a Gaussian beam
    pattern with a specified standard deviation. This beamwidth is the width
    between the two points left and right of the peak where the Gaussian
    function attains half its maximum value.

    """
    # Gaussian function reaches half its peak value at sqrt(2 log 2)*sigma
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


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

    Attributes
    ----------
    expected_width : real array, shape (2,), or float
        Initial guess of beamwidth, saved as expected width for checks
    radius_first_null : float
        Radius of first null in beam in target coordinate units (stored here for
        convenience, but not calculated internally)
    refined : int
        Number of scan-based baselines used to refine beam (0 means unrefined)
    is_valid : bool
        True if beam parameters are within reasonable ranges after fit
    std_center : array of float, shape (2,)
        Standard error of beam center, only set after :func:`fit`
    std_width : array of float, shape (2,), or float
        Standard error of beamwidth(s), only set after :func:`fit`
    std_height : float
        Standard error of beam height, only set after :func:`fit`

    """
    def __init__(self, center, width, height):
        ScatterFit.__init__(self)
        if not np.isscalar(width):
            width = np.atleast_1d(width)
        self._interp = GaussianFit(center, fwhm_to_sigma(width), height)
        self.center = self._interp.mean
        self.width = sigma_to_fwhm(self._interp.std)
        self.height = self._interp.height

        self.expected_width = width
        # Initial guess for radius of first null
        # XXX: POTENTIAL TWEAK
        self.radius_first_null = 1.3 * np.mean(self.expected_width)
        # Beam initially unrefined and invalid
        self.refined = 0
        self.is_valid = False
        self.std_center = self.std_width = self.std_height = None

    def fit(self, x, y, std_y=1.0):
        """Fit a beam pattern to data.

        The center, width and height of the fitted beam pattern (and their
        standard errors) can be obtained from the corresponding member variables
        after this is run.

        Parameters
        ----------
        x : array-like, shape (2, N)
            Sequence of 2-dimensional target coordinates (as column vectors)
        y : array-like, shape (N,)
            Sequence of corresponding total power values to fit
        std_y : float or array-like, shape (N,), optional
            Measurement error or uncertainty of `y` values, expressed as standard
            deviation in units of `y`

        """
        self._interp.fit(x, y, std_y)
        self.center = self._interp.mean
        self.width = sigma_to_fwhm(self._interp.std)
        self.height = self._interp.height
        self.std_center = self._interp.std_mean
        self.std_width = sigma_to_fwhm(self._interp.std_std)
        self.std_height = self._interp.std_height
        self.is_valid = not any(np.isnan(self.center)) and self.height > 0.
        # XXX: POTENTIAL TWEAK
        norm_width = self.width / self.expected_width
        self.is_valid &= all(norm_width > 0.9) and all(norm_width < 1.25)

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

"""Fit primary beams to receptor gains obtained at various offset pointings.

Parameters
----------
my_params : Dictionary containing telstate <class 'katdal.sensordata.TelstateToStr'>, 
        antennas, target, pointing times, track duration etc.
data_points : dict mapping receptor index to (x, y, freq, gain, weight) seq
    Complex gains per receptor, as multiple records per offset and frequency

Returns
-------
beams : dict mapping receptor name to list of :class:`BeamPatternFit`
    Fitted primary beams, per receptor and per frequency chunk

"""
def beam_fit(my_gains,NUM_CHUNKS,my_params):
    beams = {}
    # Iterate over receptors
    for a in my_gains:
        data = np.rec.fromrecords(my_gains[a], names='x,y,freq,gain,weight')
        data = data.reshape(-1, NUM_CHUNKS)
        ant = my_params['ants'][a]
        # Iterate over frequency chunks but discard typically dodgy band edges
        for chunk in range(1, NUM_CHUNKS - 1):
            chunk_data = data[:, chunk]
            is_valid = np.nonzero(~np.isnan(chunk_data['gain']) &
                                  (chunk_data['weight'] > 0.))[0]
            chunk_data = chunk_data[is_valid]
            if len(chunk_data) == 0:
                continue
            expected_width = rad2deg(ant.beamwidth * lightspeed /chunk_data['freq'][0] / ant.diameter)
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
            center_norm = beam.center / beam.std_center
            # Store beam per frequency chunk and per receptor
            beams_freq = beams.get(ant.name, [None] * NUM_CHUNKS)
            beams_freq[chunk] = beam
            beams[ant.name] = beams_freq
    return beams



"""Calculate pointing offsets per receptor based on primary beam fits.

Parameters
----------
beams : dict mapping receptor name to list of :class:`BeamPatternFit`
    Fitted primary beams, per receptor and per frequency chunk
my_params : Dictionary containing telstate <class 'katdal.sensordata.TelstateToStr'>, 
        antennas, target, pointing times, track duration etc.
Returns
-------
pointing_offsets : dict mapping receptor name to offset data (10 floats)
    Pointing offsets per receptor in degrees, stored as a sequence of
      - requested (az, el) after refraction (input to the pointing model),
      - full (az, el) offset, including contributions of existing pointing
        model, any existing adjustment and newly fitted adjustment
        (useful for fitting new pointing models as it is independent),
      - full (az, el) adjustment on top of existing pointing model,
        replacing any existing adjustment (useful for reference pointing),
      - relative (az, el) adjustment on top of existing pointing model and
        adjustment (useful for verifying reference pointing), and
      - rough uncertainty (standard deviation) of (az, el) adjustment.

"""
def calc_pointing_offsets(beams,my_params):
    import statistics
    ## Middle time for the observation
    middle_time=statistics.median(my_params['mid_times'])
    ## Atmospheric conditions at middle time of the observation
    temperature=my_params['telstate']['anc_air_temperature']
    humidity=my_params['telstate']['anc_air_relative_humidity']
    pressure=my_params['telstate']['anc_air_pressure']
    pointing_offsets = {}
    # Iterate over receptors
    for ant in sorted(my_params['ants']):
        beams_freq = my_beams.get(ant.name, [])

        beams_freq = [b for b in beams_freq if b is not None and b.is_valid]

        offsets_freq = np.array([b.center for b in beams_freq])

        offsets_freq_std = np.array([b.std_center for b in beams_freq])
        weights_freq = 1. / offsets_freq_std ** 2
        results = np.average(offsets_freq, axis=0, weights=weights_freq,returned=True)

        pointing_offset = results[0]
        pointing_offset_std = np.sqrt(1. / results[1])
        # Get existing pointing adjustment
        az_adjust = my_params['telstate'][str(ant.name)+'_pos_adjust_pointm_azim']
        el_adjust = my_params['telstate'][str(ant.name)+'_pos_adjust_pointm_elev']
        existing_adjustment = deg2rad(np.array((az_adjust, el_adjust)))
        # Start with requested (az, el) coordinates, as they apply
        # at the middle time for a moving target
        requested_azel = my_params['target'].azel(timestamp=middle_time, antenna=ant)

        # Correct for refraction, which becomes the requested value
        # at input of pointing model
        rc = RefractionCorrection()
        def refract(az, el):  # noqa: E306, E301
            """Apply refraction correction as at the middle of scan."""
            return [az, rc.apply(el, temperature, pressure, humidity)]
        refracted_azel = np.array(refract(*requested_azel))
        # More stages that apply existing pointing model and/or adjustment
        pointed_azel = np.array(ant.pointing_model.apply(*refracted_azel))

        adjusted_azel = pointed_azel + existing_adjustment
        # Convert fitted offset back to spherical (az, el) coordinates
        pointing_offset = deg2rad(np.array(pointing_offset))
        beam_center_azel = my_params['target'].plane_to_sphere(*pointing_offset,timestamp=middle_time,antenna=ant)
        # Now correct the measured (az, el) for refraction and then apply the
        # existing pointing model and adjustment to get a "raw" measured
        # (az, el) at the output of the pointing model stage
        beam_center_azel = refract(*beam_center_azel)
        beam_center_azel = ant.pointing_model.apply(*beam_center_azel)
        beam_center_azel = np.array(beam_center_azel) + existing_adjustment
        # Make sure the offset is a small angle around 0 degrees
        full_offset_azel = wrap_angle(beam_center_azel - refracted_azel)
        full_adjust_azel = wrap_angle(beam_center_azel - pointed_azel)
        relative_adjust_azel = wrap_angle(beam_center_azel - adjusted_azel)
        # Cheap 'n' cheerful way to convert cross-el uncertainty to azim form
        offset_azel_std = pointing_offset_std /             np.array([np.cos(refracted_azel[1]), 1.])
        # We store all variants of the pointing offset since we have it all
        # at our fingertips here
        point_data = np.r_[rad2deg(refracted_azel), rad2deg(full_offset_azel),
                           rad2deg(full_adjust_azel),
                           rad2deg(relative_adjust_azel), offset_azel_std]
        pointing_offsets[ant.name] = point_data
    return pointing_offsets




## Main function to run script from terminal

import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("token",help="Observation token from SARAO archive")
    parser.add_argument("--NUM_CHUNKS",default=16, help="Group the frequency channels into this many sections to obtain pointing fits")
    args=parser.parse_args()
    
    
    token=args.token
    NUM_CHUNKS=args.NUM_CHUNKS
    
    my_params=get_parameters_from_token(str(token),NUM_CHUNKS)
    my_gains=get_offset_gains(my_params,NUM_CHUNKS)
    my_beams=beam_fit(my_gains,NUM_CHUNKS,my_params)
    my_offsets=calc_pointing_offsets(my_beams,my_params)
    return my_beams
if __name__ == "__main__":
    main()

