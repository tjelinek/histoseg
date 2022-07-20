
# Histoseg - Framework for Semantic Segmentation of Histopathology Images

This project contains an implementation of a framework for multi-class histopathology image segmentation written in Python 3.8. The deep learning backbone uses Tensorflow/Keras. We use Conda for managing the packages.





## Important directories and files

*./segmentations/* is a stub; Because of the size of the images, we only 
attach segmentations sub-sampled at 1/4 of the resolution in a 
separate .zip file attached to the thesis.

*./data/* contains the data set; Because of the size of the data set, we only include the validation images. The annotations can be viewed using ASAP (see below).

*./histoseg/* contains the implementation  of the project. You can also find there the Conda environment configuration file *requirements.txt*.

*./histoseg/ml/notebooks/* contains Jupyter notebooks with implementation of the models. All the models are sub-classed from the *ModelPipeline* class.

*./miscellaneous/mapping.png* shows the color overlay we use for the individual classes.
 
 *./expert_evaluations/* contains the original expert's evaluation (in Czech) we use in the thesis.
## Installation
The project uses Anaconda. To create a new virtual environment,
use the following command.

*conda env create -f histoseg/environment.yml*

The name of the new environment will be *'histoseg'*.

Additionally, a docker image of ASAP can be used to view the annotations.
For installation, follow the instructions at https://hub.docker.com/r/vladpopovici/asap.





## Usage

For a better understanding, please look at an example notebook 
*histoseg/notebooks/Example.ipynb*

### Custom models

Class *ModelPipeline* serves as a wrapper for a model; it aims to provide a unified interface for different models' architectures.

#### load_pipeline(), save_pipeline()

