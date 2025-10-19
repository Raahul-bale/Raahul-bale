DermalScan AI: Facial Skin Aging Detection
This is a full-stack web application that uses deep learning to analyze facial images for signs of skin aging. The application is deployment-ready and uses a multi-model pipeline for comprehensive analysis.

Features
Advanced Skin Feature Analysis: Classifies images into categories (Clear Face, Wrinkles, Dark Spots, Puffy Eyes) using a custom-trained and fine-tuned ResNet50 model.

Accurate Age Estimation: Predicts the user's age using a pre-trained Caffe model, enhanced by a powerful DNN face detector.

Interactive UI: A simple and modern web interface for uploading images and viewing real-time results.

Deployment Ready: The application is configured for easy deployment on cloud platforms.

Models Used
The application's backend runs a three-stage AI pipeline:

DNN Face Detector: A deep learning model that first scans the image to find the precise location of a face.

Caffe Age Net: A pre-trained model that takes the cropped face from the detector and estimates the person's age.

Custom Skin Model (ResNet50): A powerful, fine-tuned ResNet50 model that we trained on a custom dataset to classify the overall skin condition of the image.

Project Setup
To run this project on your local machine, follow these steps:

1. Clone the Repository:

git clone [https://github.com/Raahul-bale/raahulbale_dremalscan.git](https://github.com/Raahul-bale/raahulbale_dremalscan.git)
cd raahulbale_dremalscan


2. Download the Dataset:
The dataset (~900 images) is hosted separately to keep this repository lightweight. This is only required if you want to retrain the skin feature model.

Click here to download dataset.zip

Once downloaded, unzip the file and place the dataset folder inside the main project directory.

3. Set Up the Environment:

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install the required libraries from the requirements file
pip install -r requirements.txt


4. Run the Application:

# Start the backend server
flask run


Then, open the index.html file in your browser. Using the "Live Server" extension in VS Code is recommended.

Retraining the Model
You can retrain the skin feature classification model using the provided script. After setting up the environment and placing the dataset folder, run:

python train_resnet_model.py


This will generate a new dermalscan_resnet_finetuned_model.h5 file.

This project was developed as part of the Springboard program.
