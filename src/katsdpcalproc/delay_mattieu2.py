import numpy as np


#1D direct fourier transform
def pydft(y,ws=None):
    x=np.arange(len(y))
    if ws is None:
        ws=np.roll(np.linspace(-np.pi,np.pi,len(y),endpoint=False),len(y)//2)
    Fy=np.zeros(len(ws),dtype='complex')
    for iw,w in enumerate(ws):
        Fy[iw]=np.sum(y*np.exp(-1j*x*w))
    return Fy


#uses direct fourier transform to subsample sinc function progressively near peak
#determine remaining delay and unwrap phase
def pydftdelay(signal):
    validsignal=np.nonzero(signal!=0.0)[0]
    ws=np.roll(np.linspace(-np.pi,np.pi,len(signal),endpoint=False),len(signal)//2)
    Fsignal=np.abs(np.fft.fft(signal))
    iFsignal=np.argmax(Fsignal)
    x=np.arange(len(signal))
    w0=ws[iFsignal]-(ws[1]-ws[0])/2
    w1=ws[iFsignal]+0
    w2=ws[iFsignal]+(ws[1]-ws[0])/2
    F0=np.abs(np.sum(signal*np.exp(-1j*x*w0)))#direct fourier transform at this w
    F1=Fsignal[iFsignal]
    F2=np.abs(np.sum(signal*np.exp(-1j*x*w2)))#direct fourier transform at this w
    for iterate in range(50):
        if F0>=F1 or F0>F2:
            F2=F1;w2=w1;
        else:
            F0=F1;w0=w1;
        w1=0.5*(w0+w2)
        F1=np.abs(np.sum(signal*np.exp(-1j*x*w1)))#direct fourier transform at this w
    delay_rad_per_sample=w1#radians per sample
    return delay_rad_per_sample


def mattieu2(x):
    return np.array([pydftdelay(signal) for signal in x])
