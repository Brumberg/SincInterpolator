import SincXinterpolator as SincX
import numpy as np

import matplotlib.pyplot as p

if __name__ == "__main__":
    print("Running directly")
    x = np.arange(0, 2048) / 2048
    tau = 13
    y = np.zeros(2048)
    for i in range(1, 2):
        y = y + np.sin(2 * np.pi * i * (x-tau))

    #p.plot(y)
    #p.show()
    fft_y = SincX.calculate_fft(y)
    tau = SincX.get_loc_max(tau, y)
    tau = 15
    tau = SincX.interpolate_maximum(tau, 40, fft_y)
