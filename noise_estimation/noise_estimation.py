import numpy as np
import os
import matplotlib.pyplot as plt
from skrf import Network
from statsmodels.nonparametric.smoothers_lowess import lowess

class estimateNoise:
    def smoother(self, data):
        return lowess(data, self.freq, frac=self.smoothing, return_sorted=False)

class noiseWaves(estimateNoise):
    def __init__(self, data_loc, calibrator_name, mask, T_NS=1100, spectra_number=None, monochromatic=False, smoothing_width=0.5):
        self.data_loc = data_loc
        self.cal_name = calibrator_name
        self.mask = mask
        self.spectra_number = spectra_number
        self.T_NS = T_NS

        # Load data
        self.cal_folder = f'{self.data_loc}calibration/{self.cal_name}/'

        self.Gcal = Network(f'{self.cal_folder}{self.cal_name}.s1p').s[self.mask, 0, 0]

        LNA_network = Network(f'{self.data_loc}/lna.s1p')
        self.Grec = LNA_network.s[self.mask, 0, 0]
        self.freq = LNA_network.f[mask]/1e6

        self.N_freq = len(self.freq)

        self.monochromatic = monochromatic 

        # Calculate smoothing fraction
        self.smoothing = smoothing_width/(np.max(self.freq) - np.min(self.freq))

    def smooth_PSDs(self):
        if self.spectra_number is not None:
            Pcal = np.loadtxt(f'{self.cal_folder}psd_source.txt', delimiter=',')[self.spectra_number,self.mask]
            PL = np.loadtxt(f'{self.cal_folder}psd_load.txt', delimiter=',')[self.spectra_number,self.mask]
            PNS = np.loadtxt(f'{self.cal_folder}psd_noise.txt', delimiter=',')[self.spectra_number,self.mask]
        
        else:
            Pcal = np.loadtxt(f'{self.cal_folder}psd_source.txt', delimiter=',')[self.mask]
            PL = np.loadtxt(f'{self.cal_folder}psd_load.txt', delimiter=',')[self.mask]
            PNS = np.loadtxt(f'{self.cal_folder}psd_noise.txt', delimiter=',')[self.mask]

        # Smooth the PSDs
        Pcal_smooth = self.smoother(Pcal)
        PL_smooth = self.smoother(PL)
        PNS_smooth = self.smoother(PNS)

        # Calculate the residual
        Pcal_res = Pcal - Pcal_smooth
        PL_res = PL - PL_smooth
        PNS_res = PNS - PNS_smooth

        # Calculate fractional residual
        Pcal_frac_res = Pcal_res / Pcal_smooth
        PL_frac_res = PL_res / PL_smooth
        PNS_frac_res = PNS_res / PNS_smooth

        self.Pcal = Pcal
        self.PL = PL
        self.PNS = PNS

        self.Pcal_smooth = Pcal_smooth
        self.PL_smooth = PL_smooth
        self.PNS_smooth = PNS_smooth

        self.Pcal_res = Pcal_res
        self.PL_res = PL_res
        self.PNS_res = PNS_res

        self.Pcal_frac_res = Pcal_frac_res
        self.PL_frac_res = PL_frac_res
        self.PNS_frac_res = PNS_frac_res

    def estimate(self):
        # Check if PSDs have been smoothed
        try:
            self.Pcal
        except AttributeError:
            self.smooth_PSDs()

        # Find standard deviation of the fractional residuals
        Pcal_frac_res_std = np.std(self.Pcal_frac_res)
        PL_frac_res_std = np.std(self.PL_frac_res)
        PNS_frac_res_std = np.std(self.PNS_frac_res)

        # Calculate the covariances
        sigma_Pcal = Pcal_frac_res_std * self.Pcal_smooth
        sigma_PL = PL_frac_res_std * self.PL_smooth
        sigma_PNS = PNS_frac_res_std * self.PNS_smooth

        cov_Pcal_PL = np.sum(self.Pcal_res * self.PL_res) / self.N_freq
        cov_PNS_PL = np.sum(self.PNS_res * self.PL_res) / self.N_freq

        A_res = self.Pcal_res - self.PL_res
        B_res = self.PNS_res - self.PL_res

        cov_AB = np.sum(A_res * B_res) / self.N_freq

        Q = (self.Pcal - self.PL) / (self.PNS - self.PL)

        # Calculate X_L
        XL = (np.abs(1-(self.Gcal*self.Grec))**2) / (1-np.abs(self.Gcal)**2)

        # Calculate the calibrated temperature noise
        self.noise = self.T_NS * XL / (self.PNS - self.PL) *\
                np.sqrt(sigma_Pcal**2 + sigma_PL**2 - 2 * cov_Pcal_PL +\
                        Q**2 * (sigma_PNS**2 + sigma_PL**2 - 2 * cov_PNS_PL) -\
                        2 * Q * cov_AB)
        
        # Calculate the monochromatic noise
        self.noise_mono = np.sqrt(np.mean(self.noise**2))

        if self.monochromatic:
            return self.noise_mono
        else:
            return self.noise
        
    def plot_smoothed_powers(self, output_dir):
        # Check if PSDs have been calculated
        try:
            self.Pcal
        except AttributeError:
            self.smooth_PSDs()

        # Plot PSDs
        os.makedirs(f'{output_dir}psd/', exist_ok=True)

        # Three panel plot
        fig, ax = plt.subplots(3, 3, figsize=(8, 7), dpi=300)

        # Source
        ax[0,0].plot(self.freq, self.Pcal, "k-", label='Source', zorder=1)
        ax[0,0].plot(self.freq, self.Pcal_smooth, "r-", zorder=2)

        ax[0,0].set_ylabel('PSD (arb. units)')
        ax[0,0].set_xlabel('Frequency (MHz)')

        ax[0,0].legend()

        # Residual
        ax[1,0].plot(self.freq, self.Pcal_res, "r-", zorder=1)
        ax[1,0].axhline(y=0, linestyle="--", color="k", label='Source', zorder=2)

        ax[1,0].set_ylabel('Res. PSD (arb. units)')
        ax[1,0].set_xlabel('Frequency (MHz)')

        ax[1,0].legend()

        # Fractional Residual
        ax[2,0].plot(self.freq, self.Pcal_res/self.Pcal_smooth*100, "r-", zorder=1)
        ax[2,0].axhline(y=0, linestyle="--", color="k", label='Source', zorder=2)

        ax[2,0].set_ylabel('Frac. Res. PSD (\%)')
        ax[2,0].set_xlabel('Frequency (MHz)')

        ax[2,0].legend()

        # Load
        ax[0,1].plot(self.freq, self.PL, "k-", label='Load', zorder=1)
        ax[0,1].plot(self.freq, self.PL_smooth, "r-", zorder=2)

        ax[0,1].set_ylabel('PSD (arb. units)')
        ax[0,1].set_xlabel('Frequency (MHz)')
        ax[0,1].legend()

        # Residual
        ax[1,1].plot(self.freq, self.PL_res, "r-", zorder=1)
        ax[1,1].axhline(y=0, linestyle="--", color="k", label='Load', zorder=2)

        ax[1,1].set_ylabel('Res. PSD (arb. units)')
        ax[1,1].set_xlabel('Frequency (MHz)')

        ax[1,1].legend()

        # Fractional Residual
        ax[2,1].plot(self.freq, self.PL_res/self.PL_smooth*100, "r-", zorder=1)
        ax[2,1].axhline(y=0, linestyle="--", color="k", label='Load', zorder=2)

        ax[2,1].set_ylabel('Frac. Res. PSD (\%)')
        ax[2,1].set_xlabel('Frequency (MHz)')

        ax[2,1].legend()

        # NS
        ax[0,2].plot(self.freq, self.PNS_smooth, "r-", zorder=2)
        ax[0,2].plot(self.freq, self.PNS, "k-", label='Noise', zorder=1)

        ax[0,2].set_ylabel('PSD (arb. units)')

        ax[0,2].set_xlabel('Frequency (MHz)')
        ax[0,2].legend()

        # Residual
        ax[1,2].plot(self.freq, self.PNS_res, "r-", zorder=1)
        ax[1,2].axhline(y=0, linestyle="--", color="k", label='Noise', zorder=2)

        ax[1,2].set_ylabel('Res. PSD (arb. units)')
        ax[1,2].set_xlabel('Frequency (MHz)')

        ax[1,2].legend()

        # Fractional Residual
        ax[2,2].plot(self.freq, self.PNS_res/self.PNS_smooth*100, "r-", zorder=1)
        ax[2,2].axhline(y=0, linestyle="--", color="k", label='Noise', zorder=2)

        ax[2,2].set_ylabel('Frac. Res. PSD (\%)')
        ax[2,2].set_xlabel('Frequency (MHz)')

        ax[2,2].legend()

        plt.suptitle(self.cal_name)
        plt.tight_layout()

        plt.savefig(f'{output_dir}psd/{self.cal_name}_psd.png')
        plt.close()

    def plot_noise_estimate(self, output_dir):
        # Check if noise estimate has been calculated
        try:
            self.noise
        except AttributeError:
            self.estimate()

        # Plot the calibrated temperature noise
        plt.figure(dpi=300, figsize=(5, 4))
        
        plt.plot(self.freq, self.noise, label=f'RMSE = {self.noise_mono:.2f} K', color='k')

        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Noise Estimate, $\sigma_s$ (K)')

        plt.gca().ticklabel_format(useOffset=False)

        plt.legend()
        plt.title(self.cal_name)
        plt.tight_layout()

        plt.savefig(f'{output_dir}{self.cal_name}_calibrated_noise.png')
        plt.close()
    

class noiseParameters(estimateNoise):
    def __init__(self):
        raise NotImplementedError("Noise parameters are currently unsupported")