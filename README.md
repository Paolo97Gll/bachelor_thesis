# bachelor-thesis

_Machine learning techniques for CMB glitch detection._


#### Note

The main OS used is Microsoft Windows 10 with Miniconda and Intel Distribution for Python 3: so all the paths in the code are written in the Windows style (`C:\path\to\data`). However, because the `healpy` package is compatible only with Linux and MacOS, the data cleaning is made using the Windows Subsystem for Linux (WSL) with Miniconda and Intel Distribution for Python 3: so the cleaning code use Linux style paths (`/path/to/data`).


## TODO and roadmap

- [x] **DATA CLEANING**, folder `cleaning` : clean data from various effects. Since the purpose of this thesis is to detect glitches and not to clean up the RAW signal from the galactic signal and other signals, all points that are on the galactic plane or coincide with a point source can be ignored without any consequences. The effects to be cleaned up are:

	- _Galactic dipole_ using the theorical equation.
	
	- _Galactic plane signal_ and _point sources_ using a mask extracted from the flags in SCI data.
	
	The steps to be performed are:
	
	- [x] **Mask preview**; since the SCI data follow the satellite data collection, the preview of the total mask cannot be performed starting from that data. However, the PLA provides the masks used, called `COM_Mask_PCCS-143-zoneMask_2048_R2.01` and `HFI_Mask_PointSrc_2048_R2.00`: using these masks, in HEALPix format, it's possible to have a global view of the total mask.
	
	- [x] **Clean data** by removing:
	
		- _Galactic dipole_ using the equation:
	
			<img src="http://www.sciweavers.org/tex2img.php?eq=D%28%5Cvec%7Bx%7D%2Ct%29%20%3D%20T_%5Ctext%7BCMB%7D%20%5Cleft%28%20%5Cfrac%7B1%7D%7B%20%5Cgamma%20%28t%29%20%5Cleft%28%201%20-%20%5Cvec%7B%5Cbeta%7D%28t%29%5Ccdot%5Cvec%7Bx%7D%20%5Cright%29%20%7D%20-1%20%5Cright%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="D(\vec{x},t) = T_\text{CMB} \left( \frac{1}{ \gamma (t) \left( 1 - \vec{\beta}(t)\cdot\vec{x} \right) } -1 \right)" width="322" height="54" />	
		
		- _Galactic plane signal_ and _point sources_ using the flags in SCI data. The SCI data, taken from the Planck Legacy Archive (PLA), are the so-called scientific data (already cleaned of various effects and glitches) and each data has a flag that indicates a peculiarity, i.e. point object, planet, galaxy plane and others. In particular, the flags of interest are those concerning the galactic plane and the point source:
			```
			bit 4: StrongSignal; 1 = In Galactic plane
			bit 5: StrongSource; 1 = On point source
			```
			Data with these flags must be discarded.
	
- [ ] **DATA CLASSIFICATION**, folder `classification` .

	_Number of data to be classified: 2000_ (1000 with a glitch, 1000 without it).


## Resources

- Machine learning video: [Jake VanderPlas: Machine Learning with Scikit Learn](https://www.youtube.com/watch?v=HC0J_SPm9co)

- [Plank Legacy Archive](http://pla.esac.esa.int/pla/#home)


## Documentations

`astropy` package:
- [documentation](https://docs.astropy.org/en/stable/)
- [FITS file handling](https://docs.astropy.org/en/stable/io/fits/)

`healpy` package:
- [documentation](https://healpy.readthedocs.io/en/latest/)

`scikit-learn` package:
- [documentation](https://scikit-learn.org/stable/)
