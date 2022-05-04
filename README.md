# Reproduction and Experimentation of Medical-Record-Linkage-Ensemble-Adaptation

## Introduction
This repository contains the code for the CS598 final project of Group 72 in our reproduction of Paper ID 252.

Group 72 Members: 
Steve McHenry <mchenry7@illinois.edu>, 
William Plefka <wplefka2@illinois.edu>

### Acknowledgements
This repository is an adaptation of the work originally performed by Vo et al., "Statistical supervised meta-ensemble algorithm for data linkage", published in Journal of Biomedical Informatics. Though much of the work is our own, we identify where we include or adapt the original authors' code.

The original work is publicly available online via [Journal of Biomedical Informatics](https://doi.org/10.1016/J.JBI.2019.103220).

Original Authors: 
Kha Vo <kha.vo@unsw.edu.au>,
Jitendra Jonnagaddala <jitendra.jonnagaddala@unsw.edu.au>,
Siaw-Teng Liaw <siaw@unsw.edu.au>.

The original publication's accompanying code and datasets are publicly available on the work's [github repository](https://github.com/ePBRN/Medical-Record-Linkage-Ensemble/)

### A Note on Data Privacy and Security

The patient datasets used in this project are synthetic and do not contain protected health information (PHI). Both datasets are publicly available via FEBRL and ePBRN. See the original work for details regarding these datasets.

## Requirements
In this section, we describe the software requirements for executing the model. There are no specific hardware requirements, although having approximately 2GB of memory available to the application and a multicore processor are an advantage.

### Runtime Environment
This project is written in Python, requiring Python version 3.6 or greater.

### Prerequisite Packages
The following packages are required prerequisites; all are available through popular Python package management systems such as pip. For a full list of all packages, refer to `requirements.txt` (generated by `pip freeze`)
+ `numpy` (v1.21.2)
+ `pandas` (v1.3.2)
+ `scikit-learn` (v0.24.2)
+ `recordlinkage` (v0.14)

## Executing the Ensemble Model
The model is executed by running `ensemble.py`. In this section, we describe the command line arguments and output of the model.

### Quick Start Guide
The model may be executed by simply running `ensemble.py` with no additional arguments. This default mode will execute both the reference and student model implementations for both Schemes A and B using preprocessed datasets and pretrained models. This will reproduce the models' evaluation statistics described in our report. Using the preprocessed datasets and pretrained models results in the quickest execution of the models when only a reproduction of the report's statistics is desired.

### In-Depth Guide

#### Execution
`ensemble.py` accepts several command line arguments which alter the execution of the program. The arguments are described below, and they are also accessible by running: `ensemble.py -h`

+ `-h`, `--help`  
Displays the argument summary (similar to this listing).

+ `-s`, `--scheme` with additional arguments `a` and/or `b`  
Specifies which schemes against which the model(s) will execute - A (FEBRL), B (ePBRN), or both (default).

+ `-i`, `--implementation` with additional arguments `reference` or `student`
Specified which model implementations will execute - reference (original authors'), student (students'), or both (default).

+ `-t`, `--train`  
Specifies that the models will be trained from scratch during execution rather than using a pretrained model.

+ `-d`, `--data`  
Specifies that the datasets will be reprocessed from file rather than using the preprocessed datasets.

+ `-e`, `--search`  
Specifies that the hyperparameter search specified by the authors will be performed on the selected models.

+ `-v`, `--verbose`
Specifies that the statistics output by the program will be verbose, that is, the full output used by the original authors' implementation rather than the console-friendly abbreviated style.

To execute every aspect of this program in single execution (data processing, model training, and hyperameter search), run: `ensemble.py -d -t -e`

#### Source Datasets
The source datasets are provided as files in this repository located in the root directory. Both Schemes A and B contains a training dataset file and test dataset file. The user may choose to substitute their own data files (of  the same file name and compatible format - see the dataset description section in our report) for training of and evaluation by the model. The files are:

+ Scheme A training: `febrl3_UNSW.csv`
+ Scheme A test: `febrl4_UNSW.csv`
+ Scheme B training: `ePBRN_F_dup.csv`
+ Scheme B test: `ePBRN_D_dup.csv`

#### Preprocessed Datasets and Pretrained Models
We expand on the original work by including preprocessed datasets and pretrained models. These elements are located in the `pre\` subfolder. Preprocessed datasets and pretrained models are available for both our implementation as well as the reference implementation. `ensemble.py` will automatically manage the loading and use of these elements per the user-provided training and data mode command line arguments. Using these preprocessed/pretrained elements saves measurable time during execution.

## Ensemble Model Source Files
In this section, we summarize the source files containing the code used in our reproduction.

+ `ensemble.py`  
The entry point to model execution. This script takes the input arguments and executes the models as specified by the user. This file imports all other files as necessary. Only this file needs to be executed directly as described in the execution section.
+ `FEBRL.py`  
Contains utility functions for processing of the FEBRL (Scheme A) dataset and model output evaluation.
+ `ePBRN.py`  
Contains utility functions for processing of the ePBRN (Scheme B) dataset and model output evaluation.
+ `FEBRLStudentImplementation.py`  
Contains the class definitions for our own ensemble base learner implementations for the FEBRL (Scheme A) dataset.
+ `ePBRNStudentImplementation.py`  
Contains the class definitions for our own ensemble base learner implementations for the ePBRN (Scheme B) dataset.
+ `StatusPrinter.py`  
Contains utilities for printing output neatly with timestamps.

## Appendix
The original authors provided several additional source and graphics files in their original repository which we believe serve as a helpful supplement, so we have chosen to retain them in this repository. We summarize their purpose in this section. For a more in-depth explanation, please see the original work's repository.

+ `FEBRL_UNSW_Linkage.ipynb`  
FEBRL (Scheme A) dataset loading, processing, and execution of reference and student models in Jupyter notebook form. This is the original code provided by the authors (supplemented with our own implementation). The Jupyter notebook format may be amenable to exploration and experimentation.
+ `ePBRN_UNSW_Linkage.ipynb`  
ePBRN (Scheme B) Jupyter notebook companion to the the previously-described Scheme A notebook.
+ `Plots.ipynb` and `plots\*`
Generates the graphical figures used in the original report to the plot subdirectory.
+ `Rectify_Febrl_Datasets.ipnb`  
Utilities for reproducing the FEBRL source dataset files provided in this repository.
+ `UNSW Error Generator\*`
Utilities for reproducing the ePBRN soruce dataset files provided in this repository.