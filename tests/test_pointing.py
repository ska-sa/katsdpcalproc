"Tests for pointing.py"

# These tests only use 1 polarization axis
import numpy as np
import katpoint
import pytest
from katsdpcalproc import pointing
import cmath
import random

# Creating metadata fpr tests
MIDDLE_TIME = 1691795333.43713
TEMPERATURE = 14.7
HUMIDITY = 29.1
PRESSURE = 897
TRACK_DURATION = 24
NUM_CHUNKS = 5
POLS = ["h"]

CENTRE_FREQ = 1284000000.0
BANDWIDTH = 856000000.0
N_CHANNELS = 1000

# Calculating channel and chunk freqencies
channel_freqs = CENTRE_FREQ + (np.arange(N_CHANNELS) - N_CHANNELS / 2) * (BANDWIDTH / N_CHANNELS)
chunk_freqs = channel_freqs.reshape(NUM_CHUNKS, -1).mean(axis=1)
target = katpoint.Target(
    body="J1939-6342, radec bfcal single_accumulation, 19:39:25.03, -63:42:45.6")

# Maximum distance of offset from target, in degrees
max_extent = 1.0
num_pointings = 8
# Build up sequence of pointing offsets running linearly in x and y directions
scan = np.linspace(-max_extent, max_extent, num_pointings // 2)
offsets_along_x = np.c_[scan, np.zeros_like(scan)]
offsets_along_y = np.c_[np.zeros_like(scan), scan]
offsets = np.r_[offsets_along_y, offsets_along_x]

# Creating list of antenna objects
ANT_DESCRIPTIONS = ["""m000,
        -30:42:39.8,
        21:26:38.0,
        1086.6,
        15,
        -8.264 -207 8.6 212.6 212.6 1,
        0:04:20.6 0 0:01:14.2 0:02:58.5 0:00:05.1 0:00:00.4
                          0:20:04.1 -0:00:34.5 0 0 -0:03:10.0 ,1.22""",
                    """m001,
        -30:42:39.8,
        21:26:38.0,
        1086.6,
        15,
        -8.264 -207 8.6 212.6 212.6 1,
        0:04:15.6 0 0:01:09.2 0:01:58.5 0:00:05.1 0:00:00.4
                          0:16:04.1 -0:00:34.5 0 0 -0:03:10.0 ,1.22"""]


ants = [katpoint.Antenna(ant) for ant in ANT_DESCRIPTIONS]
az_el_adjust = np.zeros((len(ants), 2))

# Creating gains, numpy array shape (no.offsets, no.polarisations, no.antennas)
weights = np.ones(10)
just_gains = []
for i in range(0, len(offsets)):
    gg = []
    for j in POLS:
        gg.append(np.array(np.ones(len(ants))))
    just_gains.append(np.array(gg))
just_gains = np.array(just_gains)


def generate_bp_gains(offsets, ants, channel_freqs, pols):
    """ Creating bp_gains, numpy array of shape:
        (no.offsets, no.freqs, no.polarisations, no.antennas)"""
    bp_gains = np.zeros((len(offsets), len(channel_freqs), len(pols), len(ants)), dtype = "complex_")
    for i in range(len(offsets)):
        for f in range(len(channel_freqs)):
            for pol in range(len(pols)):
                ex_width = []
                for a, ant in enumerate(ants):
                    # expected widths for each frequency channel
                    expected_width = pointing._voltage_beamwidths(
                        ant.beamwidth, channel_freqs[f], ant.diameter)
                    ex_width.append(expected_width)

                gains = []
                for k in ex_width:
                    new_beam = pointing.BeamPatternFit((0, 0), k, 1.0)
                    g = new_beam(x=offsets[i].T)
                    cmplx= random.uniform(-1.5, 1.5)
                    g=cmath.rect(g,cmplx)
                    gains.append(np.array(g))
                gains = np.array(gains).T
                bp_gains[i][f][pol] = gains

    return bp_gains


bp_gains = generate_bp_gains(offsets, ants, channel_freqs, POLS)
data_points = pointing.get_offset_gains(
    bp_gains,
    just_gains,
    offsets,
    ants,
    TRACK_DURATION,
    CENTRE_FREQ,
    BANDWIDTH,
    N_CHANNELS,
    POLS,
    NUM_CHUNKS)
beams = pointing.beam_fit(data_points, ants, NUM_CHUNKS)
pointing_offsets = pointing.calc_pointing_offsets(
    ants,
    MIDDLE_TIME,
    TEMPERATURE,
    HUMIDITY,
    PRESSURE,
    beams,
    target,
    az_el_adjust)


# Test that length of data_points equals the legnth of antenna list
def test_get_offset_gains_len():
    assert len(data_points) == len(ants)

# Test that incorrect shape of bp_gains will raise IncorrectShape Error


def test_get_offset_gains_shape():
    with pytest.raises(ValueError):
        pointing.get_offset_gains(
            bp_gains[0],
            just_gains[0],
            offsets,
            ants,
            TRACK_DURATION,
            CENTRE_FREQ,
            BANDWIDTH,
            N_CHANNELS,
            POLS,
            NUM_CHUNKS)

# Test legnth of each data_points element


def test_get_offset_gains_len2():
    for i in range(0, len(data_points)):
        assert len(list(data_points.items())[i][1]) == NUM_CHUNKS * len(offsets)
        for j in range(0, NUM_CHUNKS * len(offsets)):
            assert len(list(data_points.items())[i][1][j]) == 5

# Testing that the output of beam_fit are of type BeamPatternFit


def test_beam_fit_type():
    assert len(beams) == len(ants)
    for i in ants:
        assert type(beams[i.name][0]) and isinstance(beams[i.name][-1], type(None))
        for j in range(1, NUM_CHUNKS - 1):
            assert isinstance(beams[i.name][j], pointing.BeamPatternFit)

# Multiple small type errors for calc_pointing_offsets

# Test that the legnth of each pointing offset solution =10 (5 sets of (x,y) coordinates)


def test_calc_pointing_offsets_len():
    assert len(pointing_offsets) == len(ants)
    for i in range(0, len(pointing_offsets)):
        assert len(list(pointing_offsets.items())[i][1]) == 10

# Compare widths of simulated primary beams from beam_fit and original beam object


def test_fit_primary_beams():
    future_ex_widths = {}
    for a, ant in enumerate(ants):
        ex_width = []

        for chunk in range(0, NUM_CHUNKS):
            expected_width = pointing._voltage_beamwidths(
                ant.beamwidth, chunk_freqs[chunk], ant.diameter)
            ex_width.append(expected_width)
            future_ex_widths[ant.name] = ex_width

    # Feeding simulated data_points into beam_fit function
    beams = pointing.beam_fit(data_points, ants, NUM_CHUNKS)
    # Comparing output of beam_fit to the original beam object
    for chunk in range(1, NUM_CHUNKS - 1):
        for ant in beams.keys():
            assert beams[ant][chunk].expected_width == pytest.approx(
                future_ex_widths[ant][chunk], abs=0.0001)

