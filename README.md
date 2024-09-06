# Attention Models for Event Data

This repository contains the code for an internship project conducted at the **i3s, CNRS Laboratory, Sophia Antipolis, France**, as part of the **Master 2 in Data Science and AI** program at the **University of Nice Côte d'Azur**.

The project focuses on utilizing **YOLO** and **YOLO-World** models as bottom-up and top-down attention mechanisms for processing event data. The code and dataset samples provided here are designed for training and fine-tuning these models on event-based data.

## Repository Structure

1. **`dataset-example/`**
   - This directory contains an example of the dataset structure used for fine-tuning the models. Only one sample is provided here for reference. The full dataset (train, validation, and test sets) consists of:
     - **images/**: Event images generated from the [DSEC-Detection dataset](https://github.com/uzh-rpg/DSEC).
     - **labels/**: Corresponding labels for each image.
     - **prompts/**: Automatically generated descriptions of the form “This image contains...” listing the classes included in each image.

2. **`repo/`**
   - This directory includes the code for data preprocessing and model fine-tuning:
     - **`data-preprocessing/`**: Contains scripts for exploratory data analysis (EDA), handling missing values, generating prompts, and other preprocessing tasks.
     - **`finetune/`**: Includes code for fine-tuning YOLO and YOLO-World models, as well as scripts for evaluating the models after fine-tuning.

3. **`report/`**
   - This folder contains the final internship report that details the project from start to finish. Additionally, it includes two versions of papers for submission:
     - **"Cognitive Attention Models for Event Data"** (submitted to WACV 2025).
     - **"Top-Down and Bottom-Up Visual Attention in Event Data: A Survey"** (accepted for poster presentation at ICONIP Conference 2024).

## About the Project

The project leverages deep learning models to explore the effectiveness of attention mechanisms (both bottom-up and top-down) for event data, particularly focusing on the use of YOLO and YOLO-World architectures.


