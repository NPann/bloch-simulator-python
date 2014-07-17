"""
Show simulation of M(time, position, freq)
-> Try changind mode from 3 to 2 ...

"""


# Built-in modules
import numpy as np
import pylab as pl
from IPython import get_ipython

# Third-party libraries
from bloch.bloch import bloch


def msinc(n, m):
    """
    Return a hamming windowed sinc of length n, with m sinc-cycles, which means a time-bandwidth of 4xm
    """
    x = np.arange(-n/2., (n-1)/2.)/(n/2.)
    snc = np.sin(m * 2 * np.pi * x + .00001) / (m * 2 * np.pi * x + .00001)
    ms = snc * (0.54 + 0.46 * np.cos(np.pi * x))
    ms = ms * 4 * m / n
    return ms

# Setup parameters
b1 = np.hstack((np.zeros((500, )), msinc(250, 2), np.zeros((500, ))))
gr = np.hstack((np.zeros(375, ), - np.ones((125, )), np.ones((250, )), - np.ones((125, )), np.zeros((375, ))))
b1 = 1.56 * b1 / np.max(b1) / 4
dp = np.arange(-4, 4.1, .1)
T = 4e-6
df = np.arange(-250., 251., 5.)
tp = np.arange(1, b1.size + 1) * T
t1 = 1.
t2 = .2

# Run bloch simulation
[mx, my, mz] = bloch(b1, gr, tp, 1., .2, df, dp, 3)
#mxy = mx + 1j * my

# Display
f_c = 34
p_c = 40
pl.figure(figsize=(5, 9))
pl.subplot(3, 1, 1)
pl.plot(tp * 1e3, b1)
pl.xlabel('Time (ms)')
pl.ylabel('RF amplitude (Gauss)')
pl.title('RF pulse')
pl.subplot(3, 1, 2)
pl.plot(tp * 1e3, gr)
pl.xlabel('Time (ms)')
pl.ylabel('Gradient Amplitude')
pl.title('Gradient')
pl.subplot(3, 1, 3)
pl.plot(tp * 1e3, mx[f_c, p_c, :])
pl.plot(tp * 1e3, my[f_c, p_c, :])
pl.plot(tp * 1e3, mz[f_c, p_c, :])
pl.xlabel('Time (ms)')
pl.ylabel('Magnetization')
pl.title('Magnetization')
pl.tight_layout()
pl.show()
