from noise_estimation.noise_estimation import noiseWaves, noiseParameters
from skrf import Network

data_loc = '/path/to/data/'
output_dir = '/path/to/output/directory/'

cal_name = 'ant'

# Load the frequency data
freq = Network(data_loc+'/lna.s1p').f

# Mask frequencies
mask = (freq > 50e6) & (freq < 130e6)

# Plot the smoothed PSDs
estimator = noiseWaves(data_loc, cal_name, mask, spectra_number=0, monochromatic=True)
estimator.plot_smoothed_powers(output_dir)
estimator.plot_noise_estimate(output_dir)

print(f"Estimated Noise Level: {estimator.estimate()} K")