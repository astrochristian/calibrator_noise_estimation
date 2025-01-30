# Calibrator Noise Estimation

Package which allows you to estimate the expected noise level for an observation.

Given the path to a data directory this will read the PSDs and estimate the noise on the calibrated temperature using the formula in [Kirkham et al. (2025)](https://ui.adsabs.harvard.edu/abs/2024arXiv241214023K/abstract). 

This will work for calibration done with both noise waves and *noise parameters (currently not implemented)*.