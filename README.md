# CSE676-GeoLifeCLEF-2022

In this project we try to implement solutions to the [GeoLifeClef](https://www.imageclef.org/GeoLifeCLEF2022 "GeoLifeClef") challenge. The data can be found in the [kaggle page of the competition](https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9/data "kaggle page of the competition").

## Modules

We will use the [GLC](https://github.com/maximiliense/GLC "GLC") library which includes some helper functions provided by the competition itself. These helper functions help falicitate the import of image patches and observations.

## Data Import

### Method 1:

### Method 5: Ensemble Learning
The ensemble learning code is located in DL_Project_GEOLife.ipynb. Before running this file, we need to run the extract_patches.ipynb file to extract the patches for a given number of species and a given number of observations. We do this so that we can randomly create a smaller subset of the data upon which we carry out our results. We use a smaller subset because of the lack of computational resources to run our code on all of the data. 
