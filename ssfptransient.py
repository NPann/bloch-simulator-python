"""
Example of transient SSFP response calculation

"""


# Built-in modules
import numpy as np
import pylab as pl
from IPython import get_ipython

# Third-party libraries
from bloch.bloch import bloch


# Constant variable
gamma = 4258

# Setup parameters
tr = 5e-3    # (s)
trf = .1e-3  # (s) "hard" pulse
alpha = 60   # (deg)
T1 = 1.
T2 = .2
freq = np.arange(-200., 201.)  # (Hz)
N = 100
tpad = (tr - trf)/2 # (s)

# Setup B1, just delta-function RF
# Time intervals are [Tpad, Trf, Tpad]
t = np.array([tpad, trf, tpad])

# b1 is non-zeros during Trf
b1 = np.array([0,  np.pi / 180 * alpha / trf / gamma / 2 / np.pi,  0])  # (Gauss)

# Calculate steady state for comparison
[mxss, myss, mzss] = bloch(b1, 0. * b1, t, T1, T2, freq, 0., 1)
mss = 1j * myss
mss += mxss

# Start with alpha/2 , from equilibrium
[mx, my, mz] = bloch(np.array(1j * np.max(b1)/2), 0., trf, T1, T2, freq, 0, 0)

# # Repeat pulse sequence
pl.ion()
for n in range(0,N):
    [mx, my, mz] = bloch(b1, 0*b1, t, T1, T2, freq, 0, 0, mx, my, mz)

    sig = mx + 1j * my
    pl.subplot(2, 1, 1)
    pl.plot(freq, np.hstack((np.abs(sig), np.abs(mss))))
    pl.ylabel('Magnitude')
    pl.subplot(2, 1, 2)
    pl.plot(freq, np.hstack((np.angle(sig) / np.pi, np.angle(mss) / np.pi)))
    pl.ylabel('Phase / pi')
    pl.show()
    input("Press [enter] to continue.")
    pl.clf()
