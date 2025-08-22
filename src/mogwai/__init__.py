import numpy as np
import scipy.fft as sf
import matplotlib.pyplot as plt


def psd_white_1f(freq, sigma=1e-2, f_knee = 100, alpha = 2):
    # psd at 0 is 0
    psd = np.empty_like(freq)
    i_b = 0
    if freq[0] == 0:
        i_b = 1
        psd[0] = 0.0
    psd[i_b:] = (1 + np.power(f_knee/freq[i_b:],alpha ))*sigma**2
    return psd 


def generate_from_wnoise(psd, n_s, f_s):
    """Generate from white noise

    Args:
        psd (float[]): Power Spectrum Density mode 0 to Nyquist 
        n_s (int): number of sample in time serie
        f_s (float): [Herz] sampling frequency 

    Returns:
        float[n_s]: time serie with same PSD as parameter psd
    """
    w_noise = np.random.randn(n_s)
    fft_w = sf.rfft(w_noise)
    # PSD normalisation : sqrt(f_s/2)
    fft_w *= np.sqrt(psd*f_s/2)
    return sf.irfft(fft_w)

def generate_from_psd(psd, n_s,f_s):
    """Generate in Fourier space with directly PSD as module

    Args:
        psd (float[]): Power Spectrum Density mode 0 to Nyquist 
        n_s (int): number of sample in time serie
        f_s (float): [Herz] sampling frequency 

    Returns:
        float[n_s]: time serie with same PSD as parameter psd
    """
    angle = np.random.uniform(0,2*np.pi, len(psd))
    # PSD normalisation : sqrt(f_s/2)
    # SciPy FFT backward normalisation : sqrt(n_s)
    fft_c = np.exp(1j*angle)*np.sqrt(psd*(n_s*f_s/2))
    return sf.irfft(fft_c)

def generate_wf_noise(n_s,f_s):
    freq = sf.rfftfreq(n_s, 1/f_s)
    psd = psd_white_1f(freq)
    return generate_from_psd(psd, n_s,f_s)

def plot_psd(noise, f_herz, m_tt="", nperseg=1024):
    freq, psdw_noise = ss.welch(noise,  window="hann", fs=f_herz, scaling="density", nperseg=nperseg)
    plt.figure()
    plt.title(m_tt)
    # remove mode 0 and Nyquist
    plt.loglog(freq[1:-2], psdw_noise[1:-2])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD: [$U^2/Hz$]')
    plt.grid()