import numpy as np


def _newton_delay(signal, w0, epsilon, max_iters, discard_unconverged):
    delta = np.ones_like(w0)
    active = delta >= epsilon
    N = np.shape(signal)[-1]
    x = np.arange(N, dtype=float)
    njx=-1j * x
    temp = np.empty(signal.shape, dtype=np.complex128)
    w = w0.copy()
    iteration = 0
    while np.any(active) and iteration < max_iters:
        iteration += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            temp.real = 0.0
            np.outer(-w, x, temp.imag)
            np.exp(temp, temp)
            np.multiply(temp,signal,out=temp)
            X = temp.mean(axis=-1)# note objective function is actually C = X*X.conj()
            np.multiply(temp,njx,out=temp)
            dX = temp.mean(axis=-1)
            np.multiply(temp,njx,out=temp)
            d2X = temp.mean(axis=-1)
            halfdC = (X * dX.conj()).real #from dC = X*dX.conj())+X.conj()*dX = 2 * (X * dX.conj()).real
            halfd2C = (X * d2X.conj()).real + (dX * dX.conj()).real#originally d2C=(X*d2X.conj() + X.conj()*d2X + 2*dX*dX.conj()).real which is real just drop imag=0 component
            np.divide(halfdC,halfd2C,out=delta) #newton update
        delta[np.isnan(delta)] = 0.0
        active = ~np.isinf(delta) & (np.abs(delta) >= epsilon)
        w[active] = w[active] - delta[active]
        # print(iteration, (delta == 0.).sum(), np.abs(delta).max())
    if discard_unconverged:
        unconverged = np.isinf(delta) | active
        w[unconverged] = np.nan
    return w


def fft_newton_delay(signal, NFFT=None, epsilon=1e-10, max_iters=100,
               discard_unconverged=False):
    front_shape = signal.shape[:signal.ndim - 1]
    if front_shape != ():
        signal = signal.reshape(-1, signal.shape[-1])
    if NFFT is None:
        NFFT = 2*np.shape(signal)[-1]
    #get fft_peak
    absX = np.abs(np.fft.fft(signal, NFFT))
    fft_peak = absX.argmax(axis=-1)
    #get starting point freq from fft_peak
    deshift = np.where(fft_peak < NFFT // 2, fft_peak, fft_peak - NFFT)
    w0= 2. * np.pi * deshift / NFFT
    w = _newton_delay(signal, w0, epsilon, max_iters, discard_unconverged)
    if front_shape != ():
        w = w.reshape(front_shape)
    return w
