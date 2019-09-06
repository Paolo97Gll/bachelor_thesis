#!/usr/bin/env python3
# coding: utf-8


# CLEAN DATA
# Remove the galactic dust, the point sources and the galactic dipole from RAW data;
# this is necessary to make a correct classification of the glitches.


# Suppress NaturalNameWarning raised by HDFStore
import warnings
warnings.filterwarnings("ignore", category=NaturalNameWarning)


# Reading files
from astropy.io import fits
import h5py
# Scientific computing
import numpy as np
import pandas as pd
from astropy.coordinates import spherical_to_cartesian
# Plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Other
import os
import time as pytime
import subprocess


print("START CLEANING\n")

for OPERATING_DAY in range(91,106):

    if OPERATING_DAY < 100:
        OPERATING_DAY = "0" + str(OPERATING_DAY)
    else:
        OPERATING_DAY = str(OPERATING_DAY)

    print("-> Operating day:", OPERATING_DAY)

    for DETECTOR in ["143-5","143-6","143-7"]:

        print("--> Detector:", DETECTOR)


        # Model constants and parameters

        FILENAME_PTG = "D:/Tesi/data/HFI-143/HFI_TOI_143-PTG_R2.01_OD0" + OPERATING_DAY + ".fits"
        FILENAME_RAW = "D:/Tesi/data/HFI-143/HFI_TOI_143-RAW_R2.00_OD0" + OPERATING_DAY + ".fits"
        FILENAME_SCI = "D:/Tesi/data/HFI-143/HFI_TOI_143-SCI_R2.00_OD0" + OPERATING_DAY + ".fits"
        DIR = "ris"

        with fits.open(FILENAME_SCI) as f:
            T_CMB = f[DETECTOR].header["T_CMB"]
            SOLSYSDIR_ECL_COLAT_RAD = f[DETECTOR].header["ECL_THE"]
            SOLSYSDIR_ECL_LONG_RAD = f[DETECTOR].header["ECL_PHI"]
            SOLSYSSPEED_M_S = f[DETECTOR].header["SOLSPEED"]
            MAIN_LENGTH = len(f[DETECTOR].data.field("SIGNAL"))
            CALIBRATION_CONSTANT = f[DETECTOR].header["CALIB"]
            ZERO_POINT = f[DETECTOR].header["ZERO-PT"]

        # Number of elements of the average
        MA_LENGTH = 649346
        # New length
        NEW_LENGTH = MAIN_LENGTH - MA_LENGTH + 1

        SPEED_OF_LIGHT_M_S = 2.99792458e8
        PLANCK_H_MKS = 6.62606896e-34
        BOLTZMANN_K_MKS = 1.3806504e-23

        SOLSYS_SPEED_VEC_M_S = SOLSYSSPEED_M_S * np.array(
                                                          [
                                                           np.sin(SOLSYSDIR_ECL_COLAT_RAD) * np.cos(SOLSYSDIR_ECL_LONG_RAD),
                                                           np.sin(SOLSYSDIR_ECL_COLAT_RAD) * np.sin(SOLSYSDIR_ECL_LONG_RAD),
                                                           np.cos(SOLSYSDIR_ECL_COLAT_RAD),
                                                          ]
                                                         )


        # Cleaning model

        # Open SCI data and load the "FLAG" field
        with fits.open(FILENAME_SCI) as f:
            SCI_FLAG_bits = f[DETECTOR].data.field("FLAG")

        # Unpacks bits
        SCI_FLAG_bits = np.unpackbits(SCI_FLAG_bits[:, np.newaxis], axis=1)

        # Read the 4th and 5th bits
        SCI_FLAG_GD = SCI_FLAG_bits[:,3]
        SCI_FLAG_PS = SCI_FLAG_bits[:,4]

        # Sum the two masks
        MASK = SCI_FLAG_GD + SCI_FLAG_PS
        # Make "average"
        MASK = MASK[:NEW_LENGTH]

        # Every value above 0 means an unacceptable value
        MASK[MASK != 0] = 1


        # RAW data preparation

        # Open the voltages
        with fits.open(FILENAME_RAW) as f:
            obt = f["OBT"].data.field("OBT")
            data_raw = f[DETECTOR].data.field("RAW")

        # Data
        data_raw_corrected = np.array(data_raw)
        data_raw_corrected[1::2] = - data_raw_corrected[1::2]
        # Moving average
        data_ma = pd.Series(data_raw_corrected).rolling(window=MA_LENGTH).mean().iloc[MA_LENGTH-1:].values
        data = (data_raw_corrected[:NEW_LENGTH] - data_ma) / MA_LENGTH
        # Calibrate values
        data = (data - ZERO_POINT) / CALIBRATION_CONSTANT

        # Time
        # Convert the time from OBT clock to seconds and remove the offset
        time = (obt - obt[0]) / 65536
        # Take the correct number of elements
        time = time[:NEW_LENGTH]

        # Galactic dipole
        def get_dipole_temperature(directions):
            beta = SOLSYS_SPEED_VEC_M_S / SPEED_OF_LIGHT_M_S
            gamma = (1 - np.dot(beta, beta)) ** (-0.5)
            return T_CMB * (1.0 / (gamma * (1 - np.dot(directions, beta))) - 1.0)
        # Open PTG data and load the "THETA" and "PHI" fields
        with fits.open(FILENAME_PTG) as inpf:
            theta, phi = [inpf[DETECTOR].data.field(x) for x in ("THETA", "PHI")]

        # Get the directions (vectors) directly from the angular coordinates
        dir_x, dir_y, dir_z = np.array(spherical_to_cartesian(np.ones(MAIN_LENGTH),(np.pi/2-theta),(phi)))
        # Compute dipole temperature
        dipole = get_dipole_temperature(np.concatenate((dir_x[:, np.newaxis], dir_y[:, np.newaxis], dir_z[:, np.newaxis]), axis=1)[:NEW_LENGTH])


        # Remove galactic dipole signal

        # Make the regression
        sol_m, sol_q = np.linalg.lstsq(dipole[:, np.newaxis] * [1, 0] + [0, 1], data, rcond=None)[0]

        # Take the dipole out of the data
        data_final = data - (dipole * sol_m + sol_q)


        # Apply mask

        # Take out from the data the directions corresponding to the galactic dust mask
        data_cleaned = data_final[MASK == 0]
        holed_raw = data[MASK == 0]
        # Take out also on the time - this way I can have "holes" in the graph
        time_cleaned = time[MASK == 0]


        # Save data

        if not os.path.exists(DIR):
            os.makedirs(DIR)

        # Save time_cleaned and data_cleaned as Pandas DataFrame
        with pd.HDFStore(DIR + "/OUT-cleaned.h5") as out_file:
            out_file.put(OPERATING_DAY + "/" + DETECTOR, pd.DataFrame({"time_cleaned": time_cleaned, "data_cleaned": data_cleaned}, columns=["time_cleaned", "data_cleaned"]))


        # Plot data

        if not os.path.exists(DIR + "/plots"):
            os.makedirs(DIR + "/plots")

        # Galactic dipole
        index = time < 120.
        plt.plot(time[index], dipole[index])
        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [K]")
        plt.title("Dipole temperature (OD" + OPERATING_DAY + "_" + DETECTOR  + ")")
        plt.savefig(DIR + "/plots/OD" + OPERATING_DAY + "_" + DETECTOR + "-dipole_example.png", dpi=600)
        plt.close()

        # Linear regression
        index = time < 120.
        plt.plot(dipole[index], data[index])
        x_linreg = np.linspace(-0.0033, 0.003, 10)
        plt.plot(x_linreg, x_linreg * sol_m + sol_q)
        plt.ylim(-10, 10)
        plt.title("Correlation between galactic dipole and data (OD" + OPERATING_DAY + "_" + DETECTOR  + ")")
        plt.xlabel("Dipole [K]")
        plt.ylabel("Data [T_cmb V / W]")
        plt.legend(["dipole-data", "Linear regression"])
        plt.savefig(DIR + "/plots/OD" + OPERATING_DAY + "_" + DETECTOR + "-dipole_data_correlation.png", dpi=600)
        plt.close()

        # Comparison between RAW data and cleaned data
        index = time < 30.
        plt.plot(time[index], data[index], marker='.', linestyle='none', alpha=0.8, label="with dipole")
        plt.plot(time[index], data_final[index], marker='.', linestyle='none', alpha=0.8, label="without dipole")
        plt.ylim(-5,20)
        plt.title("Before and after dipole removal signal (without mask) (OD" + OPERATING_DAY + "_" + DETECTOR  + ")")
        plt.xlabel("Time [s]")
        plt.ylabel("Signal [T_cmb V / W]")
        plt.legend(["Raw data", "Cleaned data"])
        plt.savefig(DIR + "/plots/OD" + OPERATING_DAY + "_" + DETECTOR + "-first30s-data_raw_signal.png", dpi=600)
        plt.close()
        index = time_cleaned < 30.
        plt.plot(time_cleaned[index], holed_raw[index], marker='.', linestyle='none', alpha=0.8, label="with dipole")
        plt.plot(time_cleaned[index], data_cleaned[index], marker='.', linestyle='none', alpha=0.8, label="without dipole")
        plt.ylim(-5,20)
        plt.title("Before and after dipole removal signal (with mask) (OD" + OPERATING_DAY + "_" + DETECTOR  + ")")
        plt.xlabel("Time [s]")
        plt.ylabel("Signal [T_cmb V / W]")
        plt.legend(["Raw data", "Cleaned data"])
        plt.savefig(DIR + "/plots/OD" + OPERATING_DAY + "_" + DETECTOR + "-first30s-data_cleaned_signal.png", dpi=600)
        plt.close()


# Write attributes
with h5py.File(DIR + "/OUT-cleaned.h5") as f:
    f.attrs["TITLE"] = np.string_("Data cleaning output file")
    f.attrs["VERSION"] = np.string_("Date: " + pytime.asctime() + " | Script: clean_data-moving_average-ALL.py | GitHub commit ID: " + subprocess.run(["git", "log", "-1", "--format=%H"], stdout=subprocess.PIPE).stdout.decode("ASCII").rstrip())


print("\nFINISHED")


# Print a summary
with pd.HDFStore(DIR + "/OUT-cleaned.h5", "r") as out_file:
    print(out_file.info())
# Read attributes to verify
print("\nAttributes:")
with h5py.File(DIR + "/OUT-cleaned.h5", "r") as f:
    print(f.attrs["TITLE"])
    print(f.attrs["VERSION"])
