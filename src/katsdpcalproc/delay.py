import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    n = np.arange(N, dtype=np.float)
    angle = np.outer(n, freq) + phase
    x = ampl * np.exp(1j * angle.T) + np.sqrt(noise_var / 2) * noise
    # x = x / np.abs(x)
    angle_x = np.angle(x)
    absX = np.abs(np.fft.fft(x, NFFT))

    # Circular mean of phase difference
    dangle = np.diff(angle_x)
    cmpd = np.arctan2(np.sin(dangle).mean(axis=1), np.cos(dangle).mean(axis=1))
    # Lank-Reed-Pollon (LRP)
    # xx_lag1 = x[:, :-1].conj() * x[:, 1:]
    # lrp = np.angle(xx_lag1.sum(axis=1))
    # FFT (no interpolation)
    fft_peak = absX.argmax(axis=1)
    deshift = lambda k: np.where(k < NFFT // 2, k, k - NFFT)
    fft = 2. * np.pi * deshift(fft_peak) / NFFT
    # FFT (quadratic interpolation)
    peak_ind = np.c_[fft_peak - 1, fft_peak, fft_peak + 1]
    peak_val = absX[np.tile(range(K), (3, 1)).T, peak_ind % NFFT]
    vander = np.dstack((peak_ind ** 2, peak_ind, peak_ind ** 0))
    poly = np.einsum('...jk,...k', np.linalg.inv(vander), peak_val)
    fft_quad = 2. * np.pi * deshift(-0.5 * poly[:, 1] / poly[:, 0]) / NFFT
    # FFT (linear regression)
    angle_post_fft = np.angle(x * np.exp(-1j * np.outer(fft, n)))
    norm_n = (n - n.mean()) / np.dot(n - n.mean(), n - n.mean())
    leastsq_slope = lambda y: np.dot(norm_n, y.T - y.mean(axis=-1))
    fft_lsq = fft + leastsq_slope(np.unwrap(angle_post_fft))
    # FFT (secant)
    dx = -1j * x * n
    def deriv(f):
        basis = np.exp(-1j * np.outer(f, n)) / N
        X = (x * basis).sum(axis=1)
        dX = (dx * basis).sum(axis=1)
        return 2 * (X * dX.conj()).real
    def secant(left, right, epsilon):
        delta = np.ones_like(left)
        done = delta < epsilon
        f_old, f_new = left, right
        d_old, d_new = deriv(f_old), deriv(f_new)
        iter = 0
        while not all(done):
            iter += 1
            with np.errstate(divide='ignore', invalid='ignore'):
                delta = d_new * (f_new - f_old) / (d_new - d_old)
            delta[~np.isfinite(delta) | done] = 0.
            done = np.abs(delta) < epsilon
            f_old, d_old = f_new, d_new
            f_new = f_old - delta
            d_new = deriv(f_new)
            # print iter, (delta == 0.).sum(), np.abs(delta).max()
        return f_new
    left = 2. * np.pi * deshift(fft_peak - 0.5) / NFFT
    right = 2. * np.pi * deshift(fft_peak + 0.5) / NFFT
    fft_secant = secant(left, right, epsilon=1e-10)
    # Collect standard deviations
    wrap_angle = lambda th: (th + np.pi) % (2. * np.pi) - np.pi
    stdevs = [np.sqrt(crlb)]
    for freq_estm in [cmpd, fft_lsq, fft, fft_quad, fft_secant]:
        stdevs.append(wrap_angle(freq_estm - freq).std())
    return SNR_dB, 2 * np.pi / N, stdevs

fluxes = np.array([0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.])
snr = np.empty_like(fluxes)
res = np.empty_like(fluxes)
stdevs = np.empty((len(fluxes), 6))
for n, flux in enumerate(fluxes):
    snr[n], res[n], stdevs[n] = experiment(flux)
scaled_std = np.dot(np.diag(1.0 / res), stdevs)

sns.set_context("talk")
fig, ax = plt.subplots(figsize=(8, 6))
crline = ax.semilogy(np.log10(fluxes), scaled_std[:, 0], 'k--')
lines = ax.semilogy(np.log10(fluxes), scaled_std[:, 1:])
logflux = ax.xaxis.get_ticklocs()
ax.xaxis.set_ticklabels(['{:.1f}'.format(10 ** lf) for lf in logflux])
ax.legend(lines + crline,
          ('Ludwig', 'Laura', 'Lindsay', 'SKA', 'Secant', 'Best (CRLB)'))
ax.set_xlabel('Calibrator flux (Jy)')
ax.set_ylabel('Frequency standard deviation relative to 1/N')
ax.set_title('Frequency (delay) estimator performance')
fig.savefig('lightning_20160317_ludwig.pdf')

plt.show()
