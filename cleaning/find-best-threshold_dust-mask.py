#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

#-------------------------------------------------------------------------------

print("FIND BEST THRESHOLD VALUE FOR GALAXY DUST MASK")

#-------------------------------------------------------------------------------

import healpy as hp
import numpy as np
import os
import matplotlib.pyplot as plt

# THRESHOLD PARAMETERS
MIN_THRESHOLD = 0.001
MAX_THRESHOLD = 0.02
STEP = 0.0005

#-------------------------------------------------------------------------------

# Create ris directory
if not os.path.exists("ris"):
    print("\n-> Creating 'ris' directory...")
    os.makedirs("ris")

# Create find-threshold directory
if not os.path.exists("ris/find-threshold"):
    print("\n-> Creating 'ris/find-threshold' directory...")
    os.makedirs("ris/find-threshold")



# Load the HFI 353 GHz map from the FITS file downloaded from the Planck Legacy Archive (PLA)
print("\n-> Loading HFI 353 GHz map...")
hfi353 = hp.read_map("/mnt/d/Tesi/data/MilkyWay/HFI_SkyMap_353-psb_2048_R3.01_full.fits")

# Reduce the resolution of the map in order to save memory and computational time
print("\n-> Changing resolution...")
hfi353 = hp.ud_grade(hfi353, 512)

# Elaborate the mask
print("\n-> First elaboration...")

## Rotate the map from Galactic to Ecliptic coordinates
print("--> Rotate")
rotator = hp.rotator.Rotator(coord=["G","E"])
hfi353 = rotator.rotate_map_pixel(hfi353)

## Apply a smoothing filter to the map
print("--> Smoothing filter")
hfi353 = hp.smoothing(hfi353, fwhm=np.deg2rad(1.0), verbose=False)

## Normalize the pixel values
print("--> Normalize pixel values")
hfi353 -= np.min(hfi353)
hfi353 /= np.max(hfi353)

# Clip the values
print("\n-> Apply threshold")
for THRESHOLD in np.arange(MIN_THRESHOLD, MAX_THRESHOLD, STEP):
    print("--> Threshold:", THRESHOLD)
    hfi353N = np.array(hfi353)
    hfi353N[hfi353N <= THRESHOLD] = 0
    hfi353N[hfi353N > THRESHOLD] = 1
    hp.mollview(hfi353N, coord="E")
    plt.savefig("ris/find-threshold/threshold_" + str(THRESHOLD*10000) + ".png", dpi=200)
