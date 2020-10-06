# bachelor_thesis

_Machine learning techniques for glitch detection in Planck/HFI data_

## Results

See [here](https://github.com/Paolo97Gll/bachelor_thesis/blob/master/results.pdf) (file `results.pdf`).

## Description and roadmap

See [here](https://github.com/Paolo97Gll/bachelor_thesis/blob/master/thesis.pdf) (file `thesis.pdf`).

Some information about equations and followed procedures are included in the notebooks' comments.

- [x] **DATA CLEANING**, folder `cleaning`: clean data from various effects.
  
  Since the purpose of this thesis is to detect glitches and not to clean up the RAW signal from the galactic signal and other signals, all points that are on the galactic plane or coincide with a point source can be ignored without any consequences.

  The effects to be cleaned up are:

  - _Galactic dipole_ using the theoretical equation.

  - _Galactic plane signal_ and _point sources_ using a mask extracted from the flags in SCI data.

  The steps to be performed are:

  - [x] **Mask preview**; since the SCI data follow the satellite data collection, the preview of the total mask cannot be performed starting from that data. However, the [PLA](http://pla.esac.esa.int/pla/#home) provides the masks used, called `COM_Mask_PCCS-143-zoneMask_2048_R2.01` and `HFI_Mask_PointSrc_2048_R2.00`: using these masks, in [HEALPix](https://healpix.sourceforge.io/) format, it's possible to have a global view of the total mask.

  - [x] **Clean data** by removing two effects:

    - The _galactic dipole_, using the theoretical equation reported [here](https://www.aanda.org/articles/aa/abs/2014/11/aa21527-13/aa21527-13.html) (section 3.1, point 1).

    - The _galactic plane signal_ and _point sources_, using the flags in SCI data. The SCI data, taken from the Planck Legacy Archive (PLA), are the so-called scientific data (already cleaned and calibrated) and each data has a flag that indicates a peculiarity, e.g. point object, planet or galaxy plane. The flags of interest are those concerning the galactic plane and the point source:

      ```text
      bit 4: StrongSignal; 1 = In Galactic plane
      bit 5: StrongSource; 1 = On point source
      ```

      Data with these flags must be discarded.

    Cleaned data are saved in [HDF5](https://www.hdfgroup.org/) format: it's fast, light and allows you to save attributes like the title and the version of the code used.

- [x] **DATA CLASSIFICATION**, folder `classification`: classify data for the machine learning algorithm training.

  - [x] **Create code**; features:
  
    - Load and save status in a [toml](https://github.com/toml-lang/toml) formatted file, so you don't have to classify all the data at the same time.

    - Save beautiful examples.

    - Reset everything.

    As cleaned data, classified data are saved in HDF5 format, containing also attributes like OD and detector, date of classification and git commit of the script.

  - [x] **Classify data**; number of data to be classified: 2000 (1000 with a glitch, 1000 without it).

- [x] **BUILD MACHINE LEARNING MODELS**, folder `ml_models`: train and test various machine learning algorithms.

  PCA dimensionality reduction technique is used to see, in an intuitive way, if data are clustered in well-delimited groups or if they mix. Looking at the graphs, in both normal and sorted data, glitches (both single and multi) and non-glitches cluster in different and well-defined areas, while glitches and multi glitches are mixed. This means that a machine learning model can make a good distinction between glitches (both single and multi) and non-glitches. Instead, it's unlikely that a machine learning model can distinguish between glitched and multi-glitches. So, it is possible to avoid multiclass classifiers and focus only on binary classifiers. This has also been tested using the SVC model, which confirmed the deduction. So, except for the SVC model, all algorithms do not have the no-multi-glitch (nmg) - multi-glitch (mg) distinction.

  Candidate algorithms are:

  - [x] **C-Support Vector Classifier** (from scikit-learn), folder `ml_models/SVC`; in-depth descriptions of the algorithms used and why they were used are in notebooks in the model's main folder.
  
    Best scores:

    - Normal data (with mg): `0.98054 +- 0.00627` | `0.98980 +- 0.00187` (data aug, bagging)

    - Sorted data (with mg): `0.99932 +- 0.00124`

    State: **finished**.
  
  - [x] **Random Forest Classifier** (from scikit-learn), folder `ml_models/RFC`; in-depth descriptions of the algorithms used and why they were used are in notebooks in the model's main folder.
  
    Best scores:

    - Normal data (with mg): `0.91433 +- 0.01130` | `0.99608 +- 0.00118` (data aug)

    - Sorted data (with mg): `0.98992 +- 0.00518`
  
    State: **finished**.
  
  - [x] **K-Nearest Neighbors Classifier** (from scikit-learn), folder `ml_models/KNC`; in-depth descriptions of the algorithms used and why they were used are in notebooks in the model's main folder.
  
    Best scores:

    - Normal data (with mg): `0.90033 +- 0.01501` | `0.98917 +- 0.00224` (data aug)

    - Sorted data (with mg): `0.99842 +- 0.00177`
  
    State: **finished**.
  
  - [x] **Light Gradient Boosting Machine** (from lightgbm, Microsoft), folder `ml_models/LGB`; in-depth descriptions of the algorithms used and why they were used are in notebooks in the model's main folder.
  
    Best scores:

    - Normal data (with mg): `0.91316 +- 0.01207` | `0.95430 +- 0.00372` (data aug)

    - Sorted data (with mg): `0.99617 +- 0.00184`

    State: **finished**.

## Resources

- [Plank Legacy Archive](http://pla.esac.esa.int/pla/#home)

- [Jake VanderPlas: Machine Learning with Scikit Learn](https://www.youtube.com/watch?v=HC0J_SPm9co)
