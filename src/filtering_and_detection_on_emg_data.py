import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys

root_path = os.path.join(os.path.dirname(__file__), ".." )
sys.path.append(root_path)

signal_proc_path = os.path.join(root_path, "Signal_processing_toolbox/src" )
sys.path.append(signal_proc_path)

from TimeSeriesDenoising.TKEO import TKEO


def TKEO_example():
    """
    Teager kaisor operator example

    Context:
        EMG data is often noisy,
        We want to focus on a particular events that append.
    Method:
        TKEO will amplify/highly phases with high energy changes.

    .. image:: _static/images/TimeSeriesDenoising/TKEO.png

    
    """
    #%%import signal
    file_data_path = os.path.join(root_path, "data/emg4TKEO.mat")
    if not os.path.isfile(file_data_path):
        print("\n\tFile data could not be found. Please check that you have access to it\n")
        return

    emgdata= sio.loadmat(file_data_path)
    sig = emgdata["emg"].T
    time = emgdata["emgtime"]
    fs = emgdata["fs"]

    #filtering
    filtered_sig = TKEO(sig)

    #%%plot
    plt.figure()
    plt.plot(sig/max(sig), label="sig")
    plt.plot(filtered_sig/max(filtered_sig), label="filtered sig")#normalized the vector as TKEO is the square of the sig
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude or energy')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude or energy (normalised unit)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    TKEO_example()
