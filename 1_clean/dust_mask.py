#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import healpy as hp
import numpy as np
import os
import matplotlib.pyplot as plt

# Change this to find a suitable mask
THRESHOLD = 0.005

# Create plots directory
if not os.path.exists("ris"):
    print("Creating 'ris' directory...")
    os.makedirs("ris")

# Load the HFI 353 GHz map from the FITS file downloaded from the Planck Legacy Archive (PLA)
print("Loading HFI 353 GHz map...")
hfi353 = hp.read_map("../data/Milky_Way/HFI_SkyMap_353-psb_2048_R3.01_full.fits", verbose=False)

# Reduce the resolution of the map in order to save memory and computational time
print("Changing resolution...")
hfi353 = hp.ud_grade(hfi353, 1024)

# Plot galaxy map without threshold - normalized as hist
print("Drawing elliptic map...")
hp.mollview(hfi353, coord="GE", norm="hist")
plt.savefig("ris/elliptic_galaxy_map.png", dpi=800)

print("Drawing galactic map...")
hp.mollview(hfi353, norm="hist")
plt.savefig("ris/galactic_galaxy_map.png", dpi=800)


# Check if the dust mask already exists
exists = os.path.isfile("ris/HFI_dust_mask.fits.gz")
if exists:
    # If it exists, remove it
    print("-- Dust mask file already present. Il will be rewritten.")
    os.remove("ris/HFI_dust_mask.fits.gz")


print("Elaborating mask...")

# Rotate the map from Galactic to Ecliptic coordinates
rotator = hp.rotator.Rotator(coord=['G','E'])
hfi353 = rotator.rotate_map_pixel(hfi353)

# Apply a smoothing filter to the map
hfi353 = hp.smoothing(hfi353, fwhm=np.deg2rad(1.0), verbose=False)

# Normalize the pixel values
hfi353 -= np.min(hfi353)
hfi353 /= np.max(hfi353)

# Clip the values
hfi353[hfi353 <= THRESHOLD] = 0
hfi353[hfi353 > THRESHOLD] = 1

# Save the map in a new file
print("Saving mask...")
hp.write_map("ris/HFI_dust_mask.fits.gz", hfi353, coord='E')

# Load the map
print("Loading mask...")
dust_map = hp.read_map("ris/HFI_dust_mask.fits.gz", verbose=False)

# Plot galaxy mask
print("Drawing mask...")
hp.mollview(dust_map)
plt.savefig('ris/elliptic_galaxy_mask.png', dpi=800)
