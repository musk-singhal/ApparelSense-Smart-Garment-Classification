# ApparelSense: Smart Garment Classification

A comprehensive project that utilizes web scraping, deep learning model training, and Docker deployment to classify clothing images by gender and sleeve type.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project combines various technologies and techniques to create an end-to-end solution for classifying clothing images by gender and sleeve type. It involves web scraping to gather data, training a deep learning model using PyTorch, and deploying the model as a RESTful API using Flask and Docker. 

## Features

- **Web Scraping**: Scrapes product data and images from the Myntra website.
- **Deep Learning Model Training**: Trains a ResNet18 model to classify clothing images by gender and sleeve type.
- **API Deployment**: Deploys the trained model as a RESTful API using Flask and Docker.
- **Predictions**: Accepts image files via API requests and returns predictions for gender and sleeve type.
- **Scalability**: Dockerized deployment ensures easy scalability and portability.

## Requirements

- Python 3.x
- Docker

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <project_directory>
2. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Usage
1. **Data Gathering**: 
   - Run the data_gathering.py script to scrape data and images:
        ```bash
        python data_gathering.py

2. **Model Training**: 
   - Train the ResNet18 model using model_training.py:
        ```bash
        python model_training.py

3. **Model Prediction**: 
   - Start the Flask application for API deployment by running **app.py**:
        ```bash
        python app.py
   - Access the API endpoint at http://localhost:5001/classify to make predictions.


4. **Docker Deployment**: 
   - Alternatively, build the Docker image and run the container:
        ```bash
        docker build --no-cache -t flask-app .
        docker run -p 5001:5001 flask-app

## Project Structure
The project structure is as follows:

        ├── app.py
        ├── data_gathering.py
        ├── Dockerfile
        ├── main_function.py
        ├── model_prediction.py
        ├── model_training.py
        ├── README.md
        ├── requirements.txt
        └── img_dataset/                # Directory for storing scraped images
            └── ...
        └── ml_model/                   # Directory for storing trained model
            └── model_resnet18.pth
            └── model_resnet18_class_names.json
       

- `app.py`: Flask application for API deployment.
  - This is the Flask application that exposes an API endpoint /classify for making predictions. It receives a POST request with an image file, passes the image to the AutoClassification class, which in turn uses the main_function.py to perform image classification. Finally, it returns the predicted gender and sleeve type in JSON format.
- `data_gathering.py`: Script for web scraping and data collection.
  - This script is responsible for scraping data from the Myntra website and downloading images. It uses Selenium to automate web browsing and capture product information like name, brand, URL, price, and image URLs. It then downloads the images and saves them to the specified directory (img_dataset). Finally, it saves all the data to a CSV file named dataset.csv.
- `Dockerfile`: Configuration file for Docker image.
  - This file specifies the Docker image configuration. It starts from the official Python image, copies the project files into the container, installs dependencies specified in requirements.txt, and sets the command to run the Flask application (app.py).
- `main_function.py`: Main class for coordinating data gathering, model training, and prediction.
  - This script acts as a bridge between data gathering, model training, and model prediction. It defines the AutoClassification class, which has methods for data gathering (capture_data), model training (dnn_model_training), and prediction (prediction). When executed as a standalone script, it performs data gathering, model training, and prediction.
- `model_prediction.py`: Module for loading the trained model and making predictions.
  - This script loads the trained ResNet18 model and performs predictions on input images. It defines functions to load the model, load class names from the JSON file, preprocess the input image, and make predictions using the loaded model. It returns predicted gender and sleeve type for the input image.
- `model_training.py`: Module for training the ResNet18 model.
  - This script trains the ResNet18 model using PyTorch. It loads the images from the img_dataset directory, applies transformations like resizing and normalization, and splits the dataset into training and validation sets. Then it defines the ResNet18 model, sets up loss function (CrossEntropyLoss), and optimizer (SGD). It trains the model for 50 epochs, monitoring loss and accuracy. After training, it saves the trained model weights to ml_model/model_resnet18.pth and class-to-index mapping to a JSON file.
- `README.md`: Markdown file containing project documentation.
- `requirements.txt`: List of Python dependencies.
  - This file lists all the Python dependencies required for the project, including Flask, Selenium, PyTorch, and other libraries.
- `img_dataset/`: Directory for storing scraped images.
- `ml_model/`: Directory for storing trained model files. 
## Results
Learning Curve:

![WhatsApp Image 2024-04-02 at 21 09 16](https://github.com/musk-singhal/ApparelSense-Smart-Garment-Classification/assets/34962939/a74289c7-2b81-466e-9b05-357d628d1c77)
Postman API Samples:

<img width="500" alt="Screenshot 2024-04-03 at 10 05 36 PM" src="https://github.com/musk-singhal/ApparelSense-Smart-Garment-Classification/assets/34962939/b0a73f4c-e763-425c-98eb-97bd34f17fb6">
<img width="500" alt="Screenshot 2024-04-03 at 10 10 00 PM" src="https://github.com/musk-singhal/ApparelSense-Smart-Garment-Classification/assets/34962939/5f8d472d-df27-4b67-ae04-b185dc48c0b4">
<img width="500" alt="Screenshot 2024-04-03 at 10 13 20 PM" src="https://github.com/musk-singhal/ApparelSense-Smart-Garment-Classification/assets/34962939/7349e85e-17e9-4fc0-8ac3-8b939cfb2e9e">
<img width="500" alt="Screenshot 2024-04-03 at 10 15 36 PM" src="https://github.com/musk-singhal/ApparelSense-Smart-Garment-Classification/assets/34962939/8811dabd-503a-4d88-a931-98f41b1c096a">




Sample Output:

<img width="844" alt="Screenshot 2024-04-03 at 10 03 30 PM" src="https://github.com/musk-singhal/ApparelSense-Smart-Garment-Classification/assets/34962939/11be8a26-feed-4c2f-bf0e-f46a577fb2af">




## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your improvements.



