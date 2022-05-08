# GenderAndAgeRecognition
This project implements convolutional neural networks to classify the age and gender of a person from an image of their face. It uses Keras and Tensorflow for the neural network and OpenCV for processing the data and finding the face in the demo programs.

- [Dependecies](#dependencies)
- [Dataset](#dataset)
- [Usage](#usage)

# Dependencies
  - Python 3.6+

Tested on
 - Ubuntu 18.04
 - Python 3.8.13
 - Tensorflow 2.8.0
 - CUDA 11
 - OpenCV 4.5.5

# Dataset
This model uses the [UTKFace](https://github.com/aicip/UTKFace) or the [IMDB_WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) datasets. These datasets have to bo saved in the datasets/ directory in order to be processed. The processed datasets are then saved to datasets/processed_data/ as .npy files.

# Usage
 - Download this repository
 - Open Terminal/Command Prompt in the main GenderAndAgeRecognition directory
 - Install all the necessery libraries with:
```bash
pip install -r requirements.txt
```
 - run the demo program for webcam:
```bash
python webcam_demo.py
```
 - run the demo for image 
```bash
python image_demo.py <path_to_image>
```
