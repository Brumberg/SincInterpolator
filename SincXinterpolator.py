import numpy as np

def calculate_fft(signal : float):
    return np.fft.rfft(signal)

def get_loc_max(tau, signal):
    if tau >= 0 and tau < len(signal):
        tau_n = tau - 1
        tau_p = tau + 1

        while tau_p < len(signal) and signal[tau_p] > signal[tau]:
            tau = tau_p
            tau_p = tau_p + 1

        while tau_n >= 0 and signal[tau_n] > signal[tau]:
            tau = tau_n
            tau_n = tau_n - 1
    return tau


def interpolate_maximum(tau, signal_strength, signal_fft, max_step = 0.5, d = 1):
    err = 1
    signal_length = len(signal_fft)-1
    signal_length2 = 2*signal_length
    abs_max = np.max(np.abs(signal_fft))
    req_minimum = abs_max * pow(10,-signal_strength/20)/signal_length

    two_pi = 2. * np.pi
    factor = two_pi / signal_length2
    t = tau

    while np.abs(err) > 1e-12:
        grad = 0
        dgrad = 0
        for f in range(1, signal_length):
            if req_minimum*signal_length < np.abs(signal_fft[f]):
                re = np.real(signal_fft[f])/signal_length
                im = -np.imag(signal_fft[f])/signal_length

                dfactor = factor * f
                ddfactor = dfactor * dfactor

                # the phi approach
                #rect = np.sqrt(re*re+im*im)
                #phi = np.atan2(im, re)
                #cos_ = rect*np.cos(dfactor*t-phi)
                #dcos_ = -rect * dfactor * np.sin(dfactor * t - phi)
                #ddcos_ = -rect * ddfactor * np.cos(dfactor * t - phi)
                #grad = grad + dcos_
                #dgrad = dgrad + ddcos_

                # the cos sin approach
                sin_t = np.sin(dfactor * t)
                cos_t = np.cos(dfactor * t)

                dcos_ = -re * dfactor * sin_t
                ddcos_ = -re * ddfactor * cos_t

                dsin_ = im * dfactor * cos_t
                ddsin_ = -im * ddfactor * sin_t

                grad = grad + dcos_ + dsin_
                dgrad = dgrad + ddcos_ + ddsin_

        if abs(dgrad) > 1e-18:
            update = grad / dgrad
            if abs(update)>max_step:
                d = signal_length/(update*f)
            elif d < 1:
                d = min(d * 1.2, 1.)

            t = t - d*update
            err = abs(update)
        else:
            break
    return t
