#!/usr/bin/env python3
"""
Simulate a signal input to a MZM, plot output result and its frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt

import utils

plt.rcParams['font.size'] = '16'

# SECTION Configuration.
# NOTE: These value can be changed to reproduce different situations. This is just a demo.

# MZM values.
Vpi = 1           # Voltage (V) corresponding to a shift of Pi on the output phase.
bias = (3*Vpi)/2  # DC bias (V), placed at Quad+ point by default.
drift = 0.1      # Drift (V). # NOTE: Change this to see the effect on the output.
alpha = 1         # Insertion loss coefficient.

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

    frequencies_are_centered = True

    # SECTION First harmonic energy
    bin1 = round(signal_frequency / bin_size)
    if bin1 != signal_frequency / bin_size:
        frequencies_are_centered = False
        print(f"WARNING: The 1st harmonic is not centered in a bin.")

    print(f"Fundamental frequency expected at: {signal_frequency*1e6} Hz")
    print(f"Bin containing fundamental frequency is centered at: {freqs[bin1]*1e6} Hz")
    
    # Identify three points as (frequency, Log|FFT|) coordinates.
    points1 = np.array([
        [freqs[bin1-1], S_abs[bin1-1]], 
        [freqs[bin1], S_abs[bin1]], 
        [freqs[bin1+1], S_abs[bin1+1]]
        ])
    
    freq1, freq1_fft_abs = utils.parabola_estimate_peak(points1)    
    print(f"Fundamental peak found at: ({freq1}, {freq1_fft_abs}) (error_x: {abs(freq1-signal_frequency)})")
    # !SECTION

    # SECTION Second harmonic energy
    # If zero is expected, results will be affected by noise.
    bin2 = round(2*freq1 / bin_size)
    if bin2 != 2*signal_frequency / bin_size:
        frequencies_are_centered = False
        print(f"WARNING: The 2nd harmonic is not centered in the bin.")

    print(f"Second harmonic frequency expected at: {2*signal_frequency*1e6} Hz")
    print(f"Bin containing second harmonic is centered at: {freqs[bin2]*1e6} Hz")
    
    points2 = np.array([
        [freqs[bin2-1], S_abs[bin2-1]], 
        [freqs[bin2], S_abs[bin2]], 
        [freqs[bin2+1], S_abs[bin2+1]]
        ])

    freq2, freq2_fft_abs = utils.parabola_estimate_peak(points2)    
    print(f"Second harmonic peak found at: ({freq2}, {freq2_fft_abs}) (error_x: {abs(freq2-2*signal_frequency)})")
    # !SECTION
    
    # SECTION Showing ratio as indication of drift.
    print(f"Original bias drift: {drift}")
    # Inspired by (J. Svarny, 2014) paper.
    print(f"DRIFT MAGNITUDE (ratio 2nd/1st harmonic): {freq2_fft_abs/freq1_fft_abs}")

    # SECTION Relative shift calculation.
    # NOTE: This approach will not work if frequencies are not centered in the bins!!!
    if not frequencies_are_centered:
        print("WARNING: Cannot estimate drift sign, as frequencies are not centered in a bin!")
    else:    
        # NOTE: Do not use the formula below, to avoid arctan2 singularities! 
        #phaseDiff = S_arg[bin2]-2*S_arg[bin1]
        # Do the operation in complex domain first, then extract phase!
        diffValue = S[bin2]*np.conjugate(S[bin1]**2)
        phaseDiff = np.angle(diffValue) # Will be between (-pi, pi].
        print(f"Difference angle: {phaseDiff} (angular velocity {phaseDiff/(2*np.pi)})")

        if (np.abs(phaseDiff) < np.pi/2 and drift>0):
            # Drift direction is found!
            print(f"DRIFT DIRECTION: MATCHED +")
        elif (np.abs(phaseDiff) > np.pi/2 and drift<0):
            # Drift direction is found!
            print(f"DRIFT DIRECTION: MATCHED -")
        else:
            # No measurable drift direction.
            print("DRIFT DIRECTION: NOT FOUND")

    # !SECTION
    
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
    ax[2].plot(t[:num_samplesPerPeriod], mzmOut[:num_samplesPerPeriod], linestyle='-', color='purple')
    ax[2].set_ylabel('Power')

    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    ### Show frequency graph ###
    maxF = num_samples//2-1

    figFreq, axFreq = plt.subplots(nrows=2, ncols=1, sharex=True)
    figFreq.suptitle("FFT")

    axFreq[0].title.set_text('Log(|FFT|)')
    axFreq[0].plot(freqs[:maxF], S_abs_Log[:maxF], linestyle='-', color='blue')
    axFreq[0].axvline(freq1, linestyle='--', color='red')
    axFreq[0].axvline(freq2, linestyle='--', color='red')
    axFreq[0].set_ylabel('')

    axFreq[1].title.set_text('arg(FFT)')
    axFreq[1].plot(freqs[:maxF], S_arg[:maxF], linestyle='-', color='orange')
    #axFreq[1].axhline(np.angle(freq1_weighted_average), linestyle='--', color='red')
    axFreq[1].set_ylabel('')

    plt.subplots_adjust(hspace=0.5, wspace=0.2)

    # Common commands.
    plt.show()
    plt.close()