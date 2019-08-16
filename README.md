# bachelor-thesis

_Machine learning techniques for CMB glitch detection._

#### Note

The main OS used is Microsoft Windows 10 with Miniconda and Intel Distribution for Python 3: so all the paths in the code are written in the Windows style (`C:\path\to\data`). However, because the `healpy` package is compatible only with Linux and MacOS, the data cleaning is made using the Windows Subsystem for Linux (WSL) with Miniconda and Intel Distribution for Python 3: so the cleaning code use Linux style paths (`/path/to/data`).

## TODO

- [ ] DATA CLEANING; folder `cleaning`.
	- [ ] Create the **galaxy dust mask**, using healpy and a threshold value.
	- [ ] Clean data by removing:
		1) **galactic dipole**.
		2) **galaxy dust**, using both the dust mask and the sci-data bit for a double check.
		3) **sequential differences**. Data in raw files are one positive and one negative (it's a method to reduce errors), so the absolute value make the data all positive. Sequential differences divided by 2 are the mean values.
- [ ] DATA CLASSIFICATION; folder `classification`. _Number of data to be classified: 2000_ (1000 with a glitch, 1000 without it).


## Resources

- Machine learning video: [Jake VanderPlas: Machine Learning with Scikit Learn](https://www.youtube.com/watch?v=HC0J_SPm9co)

- [Plank Legacy Archive](http://pla.esac.esa.int/pla/#home)

## Documentations

Astropy package:
- [documentation](https://docs.astropy.org/en/stable/)
- [FITS file handling](https://docs.astropy.org/en/stable/io/fits/)

Healpy package:
- [documentation](https://healpy.readthedocs.io/en/latest/)
