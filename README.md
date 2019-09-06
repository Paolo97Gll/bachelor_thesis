# bachelor_thesis

_Machine learning techniques for CMB glitch detection in Planck/HFI data._

## TODO and roadmap

More information on the equations used and the procedures followed are included in the notebooks comments.

- [x] **DATA CLEANING**, folder `cleaning` : clean data from various effects. Since the purpose of this thesis is to detect glitches and not to clean up the RAW signal from the galactic signal and other signals, all points that are on the galactic plane or coincide with a point source can be ignored without any consequences. The effects to be cleaned up are:

	- _Galactic dipole_ using the theoretical equation.
	
	- _Galactic plane signal_ and _point sources_ using a mask extracted from the flags in SCI data.
	
	The steps to be performed are:
	
	- [x] **Mask preview**; since the SCI data follow the satellite data collection, the preview of the total mask cannot be performed starting from that data. However, the PLA provides the masks used, called `COM_Mask_PCCS-143-zoneMask_2048_R2.01` and `HFI_Mask_PointSrc_2048_R2.00`: using these masks, in HEALPix format, it's possible to have a global view of the total mask.
	
	- [x] **Clean data** by removing:
		- _Galactic dipole_ using the theoretical equation reported [here](https://www.aanda.org/articles/aa/abs/2014/11/aa21527-13/aa21527-13.html) (section 3.1, point 1).
		- _Galactic plane signal_ and _point sources_ using the flags in SCI data. The SCI data, taken from the Planck Legacy Archive (PLA), are the so-called scientific data (already cleaned of various effects and glitches) and each data has a flag that indicates a peculiarity, i.e. point object, planet, galaxy plane and others. In particular, the flags of interest are those concerning the galactic plane and the point source:
		
			```
			bit 4: StrongSignal; 1 = In Galactic plane
			bit 5: StrongSource; 1 = On point source
			```
			Data with these flags must be discarded.
			
	The cleaned data are saved with the [HDF5](https://www.hdfgroup.org/) format: it's faster, lighter and allow to save attributes like the title and the version of the code used.
	
- [ ] **DATA CLASSIFICATION**, folder `classification` .

	_Number of data to be classified: 2000_ (1000 with a glitch, 1000 without it).


## Resources

### Data resource

- [Plank Legacy Archive](http://pla.esac.esa.int/pla/#home)

### Machine learning

- [Jake VanderPlas: Machine Learning with Scikit Learn](https://www.youtube.com/watch?v=HC0J_SPm9co)

### Packages documentations and guides

`astropy` package:
- [documentation](https://docs.astropy.org/en/stable/)
- [FITS file handling](https://docs.astropy.org/en/stable/io/fits/)

`h5py` package:
- [documentation](http://docs.h5py.org/en/stable/)
- [How to use HDF5 files in Python](https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/)

`healpy` package:
- [documentation](https://healpy.readthedocs.io/en/latest/)

`numpy` package:
- [documentation](https://docs.scipy.org/doc/numpy/reference/)

`pandas` package:
- [documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#dataframe)
- [HDFStore](https://pandas.pydata.org/pandas-docs/stable/reference/io.html#hdfstore-pytables-hdf5)

`scikit-learn` package:
- [documentation](https://scikit-learn.org/stable/)
