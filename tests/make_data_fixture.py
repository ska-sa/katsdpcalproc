"""Fixtures for synthetic data for delay solver tests."""

import numpy as np
import pytest


def wrap_angle(th):
    return (th + np.pi) % (2.0 * np.pi) - np.pi


def calculate_params(sample_rate, n_chans, dump_period, ampl, sefd):
    bandwidth = sample_rate / 2
    channel_width = bandwidth / n_chans
    n = np.arange(n_chans, dtype=float)
    channel_freqs = bandwidth + n * channel_width
    delay_alias = 1 / channel_width
    samples = dump_period / delay_alias
    noise_var = 2 * sefd * sefd / samples
    snr = np.inf if noise_var == 0.0 else ampl * ampl / noise_var
    return channel_freqs, delay_alias, noise_var, snr


def ant_vs_baseline(n_ants):
    baselines = np.c_[np.triu_indices(n_ants, 1)]
    to_baseline = np.zeros((len(baselines), n_ants))
    for n, (ant1, ant2) in enumerate(baselines):
        to_baseline[n, ant1] = -1.0
        to_baseline[n, ant2] = +1.0
    to_ant = np.linalg.pinv(to_baseline[:, 1:])
    return baselines, to_baseline, to_ant


def generate_data(slopes, channel_freqs, ampl, noise_var,
                  window=None, tec=0, add_spike=True, rng=None, phase=None):
    if rng is None:
        rng = np.random
    n_slopes = len(slopes)
    n_chans = len(channel_freqs)
    if phase is None:
        phase = 2.0 * np.pi * rng.rand(n_slopes)
    noise = (rng.randn(n_slopes, n_chans)
             + 1j * rng.randn(n_slopes, n_chans))
    n = np.arange(n_chans, dtype=float)

    v = channel_freqs[:, np.newaxis] / 1e9
    baseline_km = 8
    iono = np.radians(0.26 * tec * baseline_km / v)

    angle = np.outer(n, slopes) + phase + iono
    x = ampl * np.exp(1j * angle.T) + np.sqrt(noise_var / 2) * noise
    if add_spike:
        x[:, n_chans // 4] += 10 * n_chans
    if window is not None:
        x *= np.atleast_2d(window)
    return x


def make_ant_data(ant_delays, sample_rate, dump_period, flux,
                  n_pols, n_chans, sefd, random):
    """Generate synthetic cross-correlations from per-antenna delays."""
    ant_delays = np.asarray(ant_delays, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        chans, delay_alias, noise_var, snr = calculate_params(
            sample_rate, n_chans, dump_period, flux, sefd)
    corrprod_lookup, to_baseline, _ = ant_vs_baseline(len(ant_delays))
    slopes_per_ant = -2.0 * np.pi * ant_delays / delay_alias
    slopes_per_bl = slopes_per_ant @ to_baseline.T
    phase_per_bl = (2.0 * np.pi * random.rand(len(ant_delays))) @ to_baseline.T
    baseline_vis = generate_data(
        slopes_per_bl, chans, ampl=flux, noise_var=noise_var, add_spike=None,
        rng=random, phase=phase_per_bl)
    data = np.repeat(baseline_vis.T[:, np.newaxis, :], n_pols, axis=1).astype(np.complex64)
    weights = np.ones_like(data, dtype=np.float32)
    return chans, corrprod_lookup, data, weights, snr, delay_alias


def cramer_rao_bound(snr, n_chans, window=None, n_ants=None):
    """Cramer-Rao bound as a standard deviation [radians/channel]."""
    if window is None:
        snr_sum = snr * n_chans
        curvature = (n_chans * n_chans - 1.0) / 12.0
    else:
        gate = window.nonzero()[0]
        weights = window / np.sum(window)
        snr_sum = snr @ weights * len(gate)
        n = np.arange(n_chans, dtype=float)
        centroid = weights @ n
        curvature = weights @ (n - centroid) ** 2
    crb = 0.5 / (snr_sum * curvature)
    if n_ants is not None:
        # Going from baseline-based to antenna-based, you score an extra factor
        # determined by np.sum((Vrt.T / s[np.newaxis, :]) ** 2, axis=1).mean()
        crb *= 2.0 / n_ants
    return np.sqrt(crb)


@pytest.fixture(name="observation_params")
def fixture_observation_params():
    """Observation parameters for synthetic delay test data."""
    return {
        "flux": 10.0,
        "sefd": 400.0,
        "dump_period": 120.0,
        "sample_rate": 1712e6,
        "N_CHANS": 4096
    }


@pytest.fixture(name="ant_delays")
def fixture_ant_delays():
    """Standard antenna delays for synthetic test data."""
    return np.array([1e-12, 8e-12, -5e-12], dtype=np.float32)


@pytest.fixture(name="make_delay_data")
def fixture_make_delay_data():
    """Build synthetic visibilities for the delay solver tests."""
    return generate_data


@pytest.fixture(name="make_ant_delay_data")
def fixture_make_ant_delay_data():
    """Provide a factory for synthetic antenna-delay visibility data."""
    return make_ant_data


@pytest.fixture(name="cramer_rao_bound")
def fixture_cramer_rao_bound():
    """Provide the Cramer-Rao delay-bound calculation helper."""
    return cramer_rao_bound


@pytest.fixture(name="calculate_params")
def fixture_calculate_params():
    """Provide the helper for deriving synthetic observation parameters."""
    return calculate_params
