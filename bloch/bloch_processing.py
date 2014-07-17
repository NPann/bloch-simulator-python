import numpy as np
import scipy as sp

#Functions to handle preprocessing for bloch simulator arguments.

def process_gradient_argument(gr):
    """
    Takes in a gradient argument and returns directional gradients.
    If gradients don't exist, returns array of zeros.
    """
    if isinstance(gr, np.ndarray):
        if 1 == len(gr.shape):
            return gr, np.zeros(gr.size), np.zeros(gr.size)
        elif 3 == gr.shape[0]:
            return gr[0], gr[1], gr[2]
        elif 2 == gr.shape[0]:
            return gr[0], gr[1], np.zeros(gr.shape[1])
        else:
            return gr[0], np.zeros(gr.shape[1]), np.zeros(gr.shape[1])
    else:
        return np.array(gr), np.array(0.),  np.array(0.)

def _times_to_intervals(endtimes, intervals, n):
    allpos = True
    lasttime = 0.0

    for val in range(n):
        intervals[val] = endtimes[val] - lasttime
        lasttime = endtimes[val]
        if intervals[val] <= 0:
            allpos = False 
    return allpos

def process_time_points(tp, rf_length):
    """
    THREE Cases:
		1) Single value given -> this is the interval length for all.
		2) List of intervals given.
		3) Monotonically INCREASING list of end times given.

	For all cases, the goal is for tp to have the intervals.
    """
    if type(tp) == type(0.0) or type(tp) == type(0):
        return tp * np.ones(rf_length)
    elif rf_length != tp.size:
        raise IndexError("time point length is not equal to rf length")
    else:
        ti = np.zeros(rf_length)
        if _times_to_intervals(tp, ti, rf_length):
            tp = ti
    return tp        

def process_off_resonance_arguments(df):
    if type(df) == type(0.0) or type(df) == type(0):
        return (df * np.ones(1)), 1 
    return df, df.size

def process_positions(dp):
    """
    Gets positions vectors if they exist. Zeros otherwise.
    """
    if isinstance(dp, (int, float)):
        return dp*np.ones(1), np.zeros(1), np.zeros(1), 1
    if 3 == dp.shape[0]:  # Assume 3 position dimensions given
        return dp[0], dp[1], dp[2], dp[0].size
    elif 2 == dp.shape[0]:  # Assume only 2 position dimensions given
        return dp[0], dp[1], np.zeros(dp.shape[1]), dp[0].size
    else:   # Either 1xN, Nx1 or something random.  In all these cases we assume that 1 position is given, because it /
            # is too much work to try to figure out anything else!
        return dp, np.zeros(dp.shape[0]), np.zeros(dp.shape[0]), dp.size

def process_magnetization(mx_0, my_0, mz_0, rf_length, freq_pos_count, mode):
    """
    Returns mx, my, and mz vectors allocated based on input parameters.
    """

    if 2 & mode:
        out_points = rf_length
    else:
        out_points = 1
    fn_out_points = out_points * freq_pos_count
    mx = np.zeros(fn_out_points)
    my = np.zeros(fn_out_points)
    mz = np.zeros(fn_out_points)
    if None != mx_0 and None != my_0 and None != mz_0:
        if isinstance(mx_0, np.ndarray) and isinstance(my_0, np.ndarray) and isinstance(mz_0, np.ndarray):
            mx_0 = mx_0.ravel()
            my_0 = my_0.ravel()
            mz_0 = mz_0.ravel()
            if mx_0.size == freq_pos_count and my_0.size == freq_pos_count and mz_0.size == freq_pos_count:
                for v in range(freq_pos_count):
                    mx[v * out_points] = mx_0[v]
                    my[v * out_points] = my_0[v]
                    mz[v * out_points] = mz_0[v]
            else:
                print('Initial magnetization is given but has wrong size, bot Npositions x Nfreq. \n')
                print('--> using [0, 0, 1] for initial magnetization \n')
                for v in range(freq_pos_count):
                    mx[v * out_points] = 0.
                    my[v * out_points] = 0.
                    mz[v * out_points] = 1.
        else:
            for v in range(freq_pos_count):
                    mx[v * out_points] = mx_0
                    my[v * out_points] = my_0
                    mz[v * out_points] = mz_0
    else:  # Init with magnetization along z
        for v in range(freq_pos_count):
            mx[v * out_points] = 0.
            my[v * out_points] = 0.
            mz[v * out_points] = 1.
    return mx, my, mz, out_points

def reshape_matrices(mx, my, mz, ntime, n_pos, nf):
    """
    Reshapes output matrices.
    """
    if ntime > 1 and nf > 1 and n_pos > 1:
        shape = (nf, n_pos, ntime)
        mx.shape = shape
        my.shape = shape
        mz.shape = shape
        return
    else:
        if ntime > 1:
            shape = ((n_pos * nf), ntime)
        else:
            shape = (nf, n_pos)
        mx.shape = shape
        my.shape = shape
        mz.shape = shape
