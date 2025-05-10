# Face Mask Detection

This project uses a deep learning model to detect whether a person is wearing a mask or not. The model is built using TensorFlow and Keras, and it is deployed in a Flask web application.
The dataset contains images of people with and without masks, and the model is trained to classify them accordingly.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Training the Model](#training-the-model)
4. [Flask Web Application](#flask-web-application)
5. [File Structure](#file-structure)
6. [License](#license)

## Installation

To run this project, you need to set up a virtual environment and install the necessary dependencies. Follow these steps:

### Step 1: Clone the repository
Clone this repository to your local machine:
```bash
git clone https://github.com/gempooo-29/face-mask-detection.git


### Step 2: Set up a virtual environment

Create a virtual environment using virtualenv or conda:

# For virtualenv
python -m venv maskenv

# For conda
conda create --name maskenv python=3.8

###Step 3: Install dependencies
Activate the virtual environment and install the required packages from requirements.txt:

# For virtualenv
maskenv\Scripts\activate

# For conda
conda activate maskenv

RUN :pip install -r requirements.txt

### Usage
To use the model or the Flask web application, follow these instructions:

Using the model
The model is saved as a .keras file and can be loaded using Keras to make predictions on images. You can load the model as follows:

from tensorflow.keras.models import load_model

model = load_model('path_to_model.keras')


### Running the Flask Web Application
To run the Flask web app, navigate to the Flask app directory and start the server:

cd MaskDetectionFlaskApp
python app.py

Visit http://127.0.0.1:5000/ in your browser to interact with the app.


Training the Model
To train the model, use the provided Jupyter Notebook train_model.ipynb. This notebook loads the dataset, processes the images, and trains the CNN model.
 After training, the model is saved as a .h5 file.

Training Steps:
1. Load and preprocess the dataset.

2. Build and compile the CNN model.

3. Train the model using the dataset.

4. Save the trained model as a .keras file.

Flask Web Application
The Flask web application allows users to upload images and get predictions on whether the person in the image is wearing a mask or not.

Folder Structure:
. MaskDetectionFlaskApp/: Contains the Flask app and necessary files.

    . app.py: Main Flask app file.

    . static/: Folder for static files (e.g., CSS, images).

    . templates/: Folder for HTML templates.

. dataset/: Contains the dataset of images used for training the model.

. train_model.ipynb: Jupyter Notebook to train the model.

. requirements.txt: List of required Python packages.

. model.keras: The trained model file.

### File Structure

. MaskDetectionFlaskApp/: The folder containing the Flask app files.

. dataset/: The folder containing the dataset used for training the model.

. train_model.ipynb: Jupyter notebook file for training the mask detection model.

. requirements.txt: The list of dependencies required for the project.

. model.keras: The saved Keras model file.





