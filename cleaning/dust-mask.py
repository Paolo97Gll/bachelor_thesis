#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

#-------------------------------------------------------------------------------

print("CREATE GALAXY DUST MASK")

#-------------------------------------------------------------------------------

import healpy as hp
import numpy as np
import os
import matplotlib.pyplot as plt

# Change this to find a suitable mask
THRESHOLD = 0.005

#-------------------------------------------------------------------------------

# Create ris directory
if not os.path.exists("ris"):
    print("\n-> Creating 'ris' directory...")
    os.makedirs("ris")

# Check if the dust mask already exists
if os.path.isfile("ris/HFI_dust_mask.fits.gz"):
    # If it exists, remove it
    answer = input("\nDust mask file already present. Overwrite? [y|n] ")
    while answer not in {"y" , "n"}:
        answer = input("Insert 'y' or 'n': ")
    if answer == "n":
        exit()



# Load the HFI 353 GHz map from the FITS file downloaded from the Planck Legacy Archive (PLA)
print("\n-> Loading HFI 353 GHz map...")
hfi353 = hp.read_map("/mnt/d/Tesi/data/MilkyWay/HFI_SkyMap_353-psb_2048_R3.01_full.fits")

# Reduce the resolution of the map in order to save memory and computational time
#print("\n--> Changing resolution...")
#hfi353 = hp.ud_grade(hfi353, 512)

# Plot galaxy map without threshold - normalized as hist
    # This method usually increases the global contrast of many images,
    # especially when the usable data of the image is represented by close
    # contrast values. Through this adjustment, the intensities can be
    # better distributed on the histogram. This allows for areas of lower
    # local contrast to gain a higher contrast. Histogram equalization
    # accomplishes this by effectively spreading out the most frequent
    # intensity values.
# Default map is in galactic coordinates
print("\n-> Drawing galactic map...")
hp.mollview(hfi353, norm="hist")
plt.savefig("ris/galactic_galaxy_map.png", dpi=800)
# Save also in ecliptic coordinates
print("-> Drawing ecliptic map...")
hp.mollview(hfi353, coord="GE", norm="hist")
plt.savefig("ris/ecliptic_galaxy_map.png", dpi=800)



# Elaborate the mask
print("\n-> Elaborating mask...")

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

## Clip the values
print("--> Make mask")
hfi353[hfi353 <= THRESHOLD] = 0
hfi353[hfi353 > THRESHOLD] = 1

## Save the map in a new file
print("--> Saving mask...")
hp.write_map("ris/HFI_DustMask.fits.gz", hfi353, coord="E")



# Load the map
print("\n-> Loading mask...")
dust_map = hp.read_map("ris/HFI_DustMask.fits.gz")

# Print galaxy mask
# Galactic coordinates
print("\n-> Drawing galactic mask...")
hp.mollview(dust_map, coord="EG")
plt.savefig("ris/galactic_galaxy_mask.png", dpi=800)
# Ecliptic coordinates
print("-> Drawing ecliptic mask...")
hp.mollview(dust_map)
plt.savefig("ris/ecliptic_galaxy_mask.png", dpi=800)
