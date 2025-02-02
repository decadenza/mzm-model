#!/usr/bin/env python3
"""
Simulate a signal input to a MZM, plot output result and its frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt

import utils

plt.rcParams['font.size'] = '16'

# SECTION Configuration.

# MZM values.
Vpi = 1           # Voltage (V) corresponding to a shift of Pi on the output phase.
bias = (3*Vpi)/2  # DC bias (V), placed at Quad+ point by default.
drift = 0.00      # Drift (V). # NOTE: Change this to see the effect on the output.

# Insertion loss coefficient.
alpha = 1

# Input signal frequency.
signal_frequency = 0.050 # MHz # NOTE: For correct phase estimation, frequency must be on a bin center!
signal_period = 1/signal_frequency # ns.
signal_intensity = 0.25 # V # NOTE: Must be small enough to stay within linear zone.

# ADC sample rate.
sampleRate = 0.512 # MHz

# Computation data.
# NOTE: The more the better to reduce boundary effects!
REPS_FACTOR = 24 # For increased resolution in frequency domain (refer to DFT definition).

# !SECTION


def ditherSignal(t):
    """ Representing the dither signal imposed over the DC bias. """

    # Adding a random shift as the model must work with any shift.
    deltaT = np.random.uniform(0, signal_period)

    return signal_intensity*np.cos(2*np.pi*signal_frequency*(t-deltaT))


def mzm(theta):
    """ Mzm transfer function as per 2010 J. Švarný paper. """
    return (1/2)*(1 + np.cos( (theta/Vpi) * np.pi)) * alpha


if __name__ == '__main__':

    # Ensure that the number of samples is a multiple of the samples per period and is a power of 2.
    # Also ensure that is an EVEN number, so that max freq is at index n/2-1 (see np.fft.fft notes).
    # We want the number of samples to be a power of 2 (radix-2 FFT).
    num_samples = utils.round_up_to_power_of_two(sampleRate*signal_period*REPS_FACTOR) # How many signal cycles are we sampling. 
    totalObservationTime = num_samples/sampleRate # ns.
    num_samplesPerPeriod = int(signal_period*sampleRate)
    
    print(f"Signal period: {signal_period} ns")
    print(f"Sampling frequency: {sampleRate} MHz")

    t = np.linspace(0, totalObservationTime, num_samples, endpoint=False, dtype=float)
    print(f"Samples per period: {num_samplesPerPeriod}")
    print(f"Total of {len(t)} samples over {totalObservationTime} ns observation time")
    
    # Input signal at the MZM (bias correction + dither)
    s = ditherSignal(t) + bias - drift

    sMin = np.min(s) # Min input value.
    sMax = np.max(s) # max input value.

    # Total input is given by signal, plus DC bias and any drift.
    mzmOut = mzm(s)  # Actual output with drift.

    rangeMin = np.min(mzmOut)
    rangeMax = np.max(mzmOut)
    print(f"MZM output range (modulation depth): [{rangeMin:0.3f}, {rangeMax:0.3f}]")
    print(f"Absolute peak difference caused by drift: {np.abs(1-rangeMin-rangeMax)}") # How much the drift affects the peak values.
    
    # Compute FFT of the MZM output.
    S = np.fft.rfft(mzmOut)

    # Extract modulus and phase.
    S_abs = np.absolute(S)

    S_abs_Log = np.log10(S_abs, where=S_abs>0) # Using Log10 for better visualisation. Excluding 0 to prevent numerical errors.
    S_arg = np.angle(S)
    
    # Info about harmonics.
    freqs = np.fft.rfftfreq(len(mzmOut), d=1/sampleRate) # Get corresponding list of frequency values.
    bin_size = sampleRate / num_samples # MHz. The width of each bin is the sampling frequency divided by the number of samples in your FFT.
    print(f"Frequency bin resolution: {1e6*bin_size:0.2f} Hz")

    # NOTE: The frequency is the center of the bin (so we must round to actually include the wanted frequency).
    # SECTION First harmonic energy
    bin1 = round(signal_frequency / bin_size)
    if bin1 != signal_frequency / bin_size:
        print(f"WARNING: The 1st harmonic is not centered in a bin.")

    print(f"Fundamental frequency expected at: {signal_frequency*1e6} Hz")
    print(f"Bin containing fundamental frequency is centered at: {freqs[bin1]*1e6} Hz")
    
    
    
    ### Show the time graph ###
    figTime, ax = plt.subplots(nrows=3, ncols=1)
    figTime.suptitle("Time")
    
    # Showing just one period.
    ax[0].title.set_text('Input signal')
    ax[0].plot(t[:num_samplesPerPeriod], s[:num_samplesPerPeriod], linestyle='-', color='red')
    ax[0].set_ylabel('Voltage')

    mzmObsWindow = np.linspace(0, 2*Vpi, 100, endpoint=False, dtype=float)
    ax[1].title.set_text('MZM transfer function')
    ax[1].vlines([bias, bias-drift], 0, 1, colors=["red", "black"])
    ax[1].vlines([sMin, sMax], 0, 1, colors="black", linestyles="dotted")

    ax[1].plot(mzmObsWindow, mzm(mzmObsWindow), label="V", linestyle='-', color='green')
    ax[1].set_ylabel('Power')

    ax[2].title.set_text('MZM output')
    ax[2].plot(t[:num_samplesPerPeriod], mzmOut[:num_samplesPerPeriod], linestyle='-', color='blue')
    ax[2].set_ylabel('Power')

    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    ### Show frequency graph ###
    maxF = num_samples//2-1

    figFreq, axFreq = plt.subplots(nrows=2, ncols=1, sharex=True)
    figFreq.suptitle("FFT")

    axFreq[0].title.set_text('Log(|FFT|)')
    axFreq[0].plot(freqs[:maxF], S_abs_Log[:maxF], linestyle='-', color='purple')
    axFreq[0].set_ylabel('')

    axFreq[1].title.set_text('arg(FFT)')
    axFreq[1].plot(freqs[:maxF], S_arg[:maxF], linestyle='-', color='orange')
    #axFreq[1].axhline(np.angle(freq1_weighted_average), linestyle='--', color='red')
    axFreq[1].set_ylabel('')

    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    # Common commands.
    plt.show()
    plt.close()