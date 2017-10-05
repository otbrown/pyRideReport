# helpers.py
# helper functions for the RideReport class
# Oliver Thomson Brown
# 2017-10-02

import numpy as np

def movavg(data, window):
    '''
    Calculates a moving average using a particular window size.
    The window is truncated when it meets the edge of the data.

    data is a 1d numpy array, window is an integer
    if window is odd, then it defines a region centred on the current
    index, over which the values are averaged
    if window is even it defines the same region as window + 1 would
    '''
    N = len(data)
    window_buffer = np.floor(window / 2).astype(int)
    smooth_data = data
    for index in range(0, window_buffer):
        start = 0
        end = index + window_buffer + 1
        smooth_data[index] = data[start:end].mean()
    for index in range(window_buffer, N - window_buffer):
        start = index - window_buffer
        end = index + window_buffer + 1
        smooth_data[index] = data[start:end].mean()
    for index in range(N - window_buffer, N - 1):
        start = index - window_buffer
        end = N - 1
        smooth_data[index] = data[start:end].mean()

    return smooth_data

def tih_calc(brevet_distance, brevet_speed, ride_distance, elapsed_time):
        SECONDS_IN_HOUR = 60 * 60
        METRES_IN_KM = 1000

        time_limit = SECONDS_IN_HOUR * brevet_distance / brevet_speed
        delta_time = time_limit - elapsed_time
        delta_distance = ((ride_distance[-1] - ride_distance)
                          / METRES_IN_KM)
        time_in_hand = ( delta_time - (SECONDS_IN_HOUR *
                        (delta_distance / brevet_speed)) )
        return time_in_hand
