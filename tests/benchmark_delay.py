
# Delay estimator comparison for lightning talk.
#
# Ludwig Schwardt
# 16 March 2016
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from katsdpcalproc.delay import (mean_phase_diff, fft_coarse, fft_quadratic,
                              fft_leastsq, fft_secant)


def experiment(flux=10., SEFD=400., dump_period=2.0, channels=4096,
               sample_rate=1712e6, repeats=1000, fft_factor=2):
    N, K, ampl = channels, repeats, flux

    NFFT = fft_factor * N
    samples = dump_period * sample_rate / (2 * N)
    noise_var = 2 * SEFD * SEFD / samples
    SNR = ampl * ampl / noise_var
    SNR_dB = 10.0 * np.log10(SNR)

    freq = 2. * np.pi * (np.random.rand(K) - 0.5)
    freq = 0.99 * freq
    phase = 2. * np.pi * np.random.rand(K)
    noise = np.random.randn(K, N) + 1j * np.random.randn(K, N)
    crlb = 6.0 / (SNR * N * (N * N - 1.0))

    n = np.arange(N, dtype=float)
    angle = np.outer(n, freq) + phase
    x = ampl * np.exp(1j * angle.T) + np.sqrt(noise_var / 2) * noise

    # Circular mean of phase difference
    cmpd = mean_phase_diff(x)
    # FFT (no interpolation)
    fft = fft_coarse(x, NFFT)
    # FFT (quadratic interpolation)
    fft_quad = fft_quadratic(x, NFFT)
    # FFT (linear regression)
    fft_lsq = fft_leastsq(x, NFFT)
    # FFT (secant)
    fft_sec = fft_secant(x, NFFT)
    # Collect standard deviations
    wrap_angle = lambda th: (th + np.pi) % (2. * np.pi) - np.pi
    stdevs = [np.sqrt(crlb)]
    for freq_estm in [cmpd, fft_lsq, fft, fft_quad, fft_sec]:
        stdevs.append(wrap_angle(freq_estm - freq).std())
    return SNR_dB, 2 * np.pi / N, stdevs


fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
snr = np.empty_like(fluxes)
res = np.empty_like(fluxes)
stdevs = np.empty((len(fluxes), 6))
for n, flux in enumerate(fluxes):
    snr[n], res[n], stdevs[n] = experiment(flux=flux)
scaled_std = np.dot(np.diag(1.0 / res), stdevs)

sns.set_context("talk")

fig, ax = plt.subplots(figsize=(8, 6))
log_fluxes = np.log10(fluxes)
crline = ax.semilogy(log_fluxes, scaled_std[:, 0], 'k--', marker='o')
lines = ax.semilogy(log_fluxes, scaled_std[:, 1:], marker='.')
ax.xaxis.set_ticks(log_fluxes)
ax.xaxis.set_ticklabels(['{:g}'.format(fl) for fl in fluxes])
ax.set_xlim(log_fluxes[0], log_fluxes[-1])
ax.legend(lines + crline,
          ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant', 'Best (CRLB)'))
ax.set_xlabel('Calibrator flux (Jy)')
ax.set_ylabel('Frequency standard deviation relative to 1/N')
ax.set_title('Frequency (delay) estimator performance')
fig.savefig('delay_estm_vs_flux.pdf')

log_sizes = np.arange(7, 14)
snr = np.empty(len(log_sizes))
res = np.empty(len(log_sizes))
stdevs = np.empty((len(log_sizes), 6))
for n, log_size in enumerate(log_sizes):
    N = 2 ** log_size
    snr[n], res[n], stdevs[n] = experiment(dump_period=2.0 * N / 4096, channels=N)
scaled_std = np.dot(np.diag(1.0 / res), stdevs)

fig, ax = plt.subplots(figsize=(8, 6))
crline = ax.semilogy(log_sizes, scaled_std[:, 0], 'k--', marker='o')
lines = ax.semilogy(log_sizes, scaled_std[:, 1:], marker='.')
ax.xaxis.set_ticks(log_sizes)
ax.xaxis.set_ticklabels(['{:g}'.format(2 ** ls) for ls in log_sizes])
ax.set_xlim(log_sizes[0], log_sizes[-1])
ax.legend(lines + crline,
          ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant', 'Best (CRLB)'))
ax.set_xlabel('Number of samples (N)')
ax.set_ylabel('Frequency standard deviation relative to 1/N')
ax.set_title('Frequency (delay) estimator performance')
fig.savefig('delay_estm_vs_N.pdf')

plt.show()
