#!/usr/bin/env python3
# coding: utf-8


# CLEAN DATA
# Remove the galactic dust, the point sources and the galactic dipole from RAW data; this is necessary to make a correct classification of the glitches.


import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import statistics as sts
from scipy.linalg import lstsq


for OPERATING_DAY in range(91,106):

    if OPERATING_DAY < 100:
        OPERATING_DAY = "0" + str(OPERATING_DAY)
    else:
        OPERATING_DAY = str(OPERATING_DAY)

    print("-> Operating day:", OPERATING_DAY)

    for DETECTOR in ["143-5","143-6","143-7"]:

        print("--> Detector:", DETECTOR)


        # Model constants and parameters

        FILENAME_PTG = "/mnt/d/Tesi/data/HFI-143/HFI_TOI_143-PTG_R2.01_OD0" + OPERATING_DAY + ".fits"
        FILENAME_RAW = "/mnt/d/Tesi/data/HFI-143/HFI_TOI_143-RAW_R2.00_OD0" + OPERATING_DAY + ".fits"
        FILENAME_SCI = "/mnt/d/Tesi/data/HFI-143/HFI_TOI_143-SCI_R2.00_OD0" + OPERATING_DAY + ".fits"
        DIR = "ris/OD" + OPERATING_DAY + "_" + DETECTOR

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
        # Moving average
        data_ma = pd.Series(data_raw).rolling(window=MA_LENGTH).mean().iloc[MA_LENGTH-1:].values
        data = (data_raw[:NEW_LENGTH] - data_ma) / MA_LENGTH
        data = np.abs(data)
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
        directions = hp.ang2vec(theta, phi)[:NEW_LENGTH]
        # Compute dipole temperature
        dipole = get_dipole_temperature(directions)


        # Remove galactic dipole signal

        # Calculate the medians for the data and the dipole temperatures
        median_data = sts.median(data)
        median_dipole = sts.median(dipole)
        # Rescale accordingly
        data = data - median_data
        dipole = dipole - median_dipole

        # Evaluate the G factor
        M = dipole[:, np.newaxis]*[0, 1]
        p, res, rnk, s = lstsq(M, data)
        G = p[1]

        # Take the dipole out of the data
        data_final = data - G * dipole


        # Apply mask

        # Take out from the data the directions corresponding to the galactic dust mask
        data_cleaned = data_final[MASK == 0]
        holed_raw = data[MASK == 0]
        # Take out also on the time - this way I can have "holes" in the graph
        time_cleaned = time[MASK == 0]


        # Save data

        if not os.path.exists(DIR):
            os.makedirs(DIR)

        np.savetxt(DIR + "/FINAL-cleaned_time.txt", time_cleaned)
        np.savetxt(DIR + "/FINAL-cleaned_data.txt", data_cleaned)


        # Plot data

        if not os.path.exists(DIR + "/plots"):
            os.makedirs(DIR + "/plots")

        # Comparison between RAW data and cleaned data - first `30s`

        index = time < 30.
        plt.plot(time[index], data[index], marker='.', linestyle='none', alpha=0.9, label="with dipole")
        plt.plot(time[index], data_final[index], marker='.', linestyle='none', alpha=0.9, label="without dipole")
        #plt.plot(time[index], dipole[index]*10**-4)
        plt.title("Before and after dipole removal signal (without mask)")
        plt.xlabel("Time [s]")
        plt.ylabel("Signal [V]")
        plt.legend(["Raw data", "Cleaned data"])
        axes = plt.gca()
        axes.set_ylim([-5,25])
        plt.savefig(DIR + "/plots/first-data_raw_signal.png", dpi=600)
        plt.close()

        index = time_cleaned < 30.
        plt.plot(time_cleaned[index], holed_raw[index], marker='.', linestyle='none',  alpha=0.9, label="with dipole")
        plt.plot(time_cleaned[index], data_cleaned[index], marker='.', linestyle='none', alpha=0.9, label="without dipole")
        plt.title("Before and after dipole removal signal (with mask)")
        plt.xlabel("Time [s]")
        plt.ylabel("Signal [V]")
        plt.legend(["Raw data", "Cleaned data"])
        axes = plt.gca()
        axes.set_ylim([-5,25])
        plt.savefig(DIR + "/plots/first-data_cleaned_signal.png", dpi=600)
        plt.close()
