# Dataset

This folder contains the files for the preprocessing steps and the dataset creation step for training and testing the models.

- Dataset creation step for each task has been described (with code) in separate .py files (the file are accordingly named).

- Our proposed patch selection method has been embedded with the 'create_dataset_solar_seg.py' file, as this method was only applied for Solar Panel Segmentation Task.

- The inputs required by the functions in these files are the file paths. For 'create_dataset_solar_seg.py' file, two additional parameters are required, specifically for the patch selection method. 
Please go throught the file to know their functions and their expected value.

- The data prerpocessing steps are mentioned in 'preprocess.py' file.
