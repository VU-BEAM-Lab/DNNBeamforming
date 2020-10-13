# DNN Beamforming Example

## TABLE OF CONTENTS

1. [Overview](#Overview)
2. [Setup](#Setup)
3. [Running the Code](#Running-the-Code)
4. [License](#License)
5. [Citing](#Citing)
6. [Acknowledgements](#Acknowledgements)

## OVERVIEW

This repository includes example source code for training a deep neural network (DNN) regression-based beamformer, as described in the following work:

[1] Luchies, A. C., and B. C. Byram. "Deep neural networks for ultrasound beamforming." IEEE TMI 37.9 (2018): 2010-2021.

The networks proposed in [1] operate on frequency domain data. The networks used in this repository operate on time domain data. The general workflow is described below and in the included PDF: supplementary.pdf

## SETUP

1 - Download anaconda here:
  - https://www.anaconda.com/products/individual

2 - Set up virtual environment and install necessary packages. For example, via the anaconda powershell prompt on a Windows without a GPU you can use the following commands:
  - conda create --name aium
  - conda activate aium
  - conda install numpy scipy h5py matplotlib pandas jupyter
  - conda install pytorch torchvision cpuonly -c pytorch
  - pip install git+git://github.com/stared/livelossplot.git
  
3 - Close out of current terminal or powershell prompt and open a new one
  - This is to complete the livelossplot installation
  - Make sure to activate the aium environment once in the new terminal or prompt

4 - Run jupyter notebook from within the folder containing the training and evaluation scripts. For example, via the anaconda powershell prompt, you can launch a jupyter notebook by typing the following:
  - jupyter notebook

IMPORTANT NOTE: Go to https://pytorch.org/ to edit the fourth bullet in step 2 according to your system and GPU availability.

## RUNNING THE CODE

1 - Make training data from time delayed RF channel data.
  - NOTE: These data are already made for the purposes of this demo. 
  - Example matlab code to make these data are included in the train_data folder.
  - This code is specific to physical phantom anechoic cysts.

2 - Split up training data into training and validation sets.
  - Make json file using train_data/json/phantom10mm.py. 
  - NOTE: This file has already been generated for the purposes of this demo.
  - This file indicates which data we are using for training vs. training evaluation vs. validation. 

3 - Train a network using provided jupyter notebook or python script (train_network_example).
  - Necessary python packages are indicated at the top of each script.
  - It is recommended to set up a virtual environment with only the necessary packages installed.
  - There is a 'DEFINE MODEL PARAMS' section in the training scripts which the user can edit to specify output file information and adjust various hyperparameters.

4 - Evaluate network performance using the provided jupyter notebook or python script (evaluate_network_example)
  - Necessary python packages are indicated at the top of each script.
  - It is recommended to set up a virtual environment with only the necessary packages installed.
  - The user should specify which model and test data to evaluate: i.e., change the model_path, test_data_path, and test_data_name variables as necessary
  - Example physical phantom and in vivo data are in the test_data folder. *NOTE: These mat files are saved as -v7.3 for h5 loading in python.
  
## LICENSE

Copyright 2020 Jaime Tierney, Adam Luchies, and Brett Byram

These materials are made available under the Apache License, Version 2.0. For details, refer to the LICENSE file.

## CITING

Please cite the below manuscripts when using any of the code or data within this repository. 

[1] - Luchies, A. C., and B. C. Byram. "Deep neural networks for ultrasound beamforming." IEEE TMI 37.9 (2018): 2010-2021.

[2] - Luchies, Adam C., and Brett C. Byram. "Training improvements for ultrasound beamforming with deep neural networks." Physics in Medicine & Biology 64.4 (2019): 045018.

## ACKNOWLEDGEMENTS

This work was supported in part by NIH grants R01EB020040 and S10OD016216-01, NSF grant IIS-175099, and by the Data Science Institute at Vanderbilt University.
