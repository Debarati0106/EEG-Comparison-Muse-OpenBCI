# EEG-Comparison-Muse-Upsidedown-Labs
Developed an EEG comparison pipeline using MATLAB and Python to evaluate OpenBCI and Muse recordings through signal preprocessing, Welch PSD, frequency-band power analysis, sampling continuity checks, channel validation, and 50 Hz line-noise assessment.

# Comparative EEG Signal Analysis: OpenBCI vs Muse
This project focuses on the comparative analysis of EEG recordings acquired using the OpenBCI and Muse EEG systems. The objective is to examine signal quality, frequency-domain characteristics, noise contamination, and recording integrity using a structured signal-processing workflow implemented with MATLAB and Python.

The analysis begins with inspecting the raw recordings, verifying sampling information, checking timestamp continuity, examining channel configurations, and identifying inactive or zero-valued channels. The OpenBCI recording includes active Fp2 and O1 channels alongside unused channels containing zero values, while the Muse recording provides its own channel configuration for comparison.

Both recordings undergo bandpass filtering within the 0.5–40 Hz range to retain relevant EEG activity. Frequency-domain analysis is performed using Welch's Power Spectral Density (PSD) method with a Hann window, a four-second window length, and 50% overlap, providing an approximate frequency resolution of 0.25 Hz at the analysed sampling rate.

The pipeline evaluates power across standard EEG frequency bands, including delta, theta, alpha, beta, and gamma. Additional analysis includes raw signal visualization, PSD comparisons, spectral characteristics, and 50 Hz line-noise assessment. Particular attention is given to ensuring that active channels exhibit meaningful spectral power and that plots use appropriate amplitude and power units.

The analysis identified measurable differences in line-noise ratios between the recordings. For the O1 channel, the estimated 50 Hz line-noise ratios were approximately 0.335% for OpenBCI and 0.075% for Muse. These findings describe the specific recordings analysed and do not establish universal superiority of either EEG system.

This project demonstrates practical applications of digital signal processing, EEG preprocessing, spectral analysis, channel validation, and comparative neurotechnology evaluation.

Technologies:MATLAB, Python, OpenBCI, Muse EEG, Welch PSD, Digital Signal Processing, and EEG Signal Quality Assessment.
