# Heart Disease Prediction using Random Forest Classifier and Flask API

## Project Overview

This project showcases the end-to-end process of developing a data science solution to predict heart disease, utilizing machine learning and web development technologies. The core of this project includes:

1. **Data Collection & Preprocessing**: Gathering and preparing the dataset for analysis.
2. **Model Development**: Building and training a predictive model using a Random Forest Classifier.
3. **Model Evaluation**: Assessing the performance of the model using various metrics.
4. **Model Deployment**: Deploying the trained model using Flask to provide real-time predictions through a web application.

The goal of this project is to build a functional web app that allows users to input various health parameters and predict whether they are at risk for heart disease.

## Dataset Description

The dataset used in this project includes several health-related features to predict the presence of heart disease. The dataset consists of **13 input features** and one **target variable** that indicates whether a person has heart disease (1) or not (0).

### Features:

- **age**: Age of the person.
- **sex**: Gender (1 = male, 0 = female).
- **cp**: Chest pain type (values from 0 to 3).
- **trestbps**: Resting blood pressure (in mm Hg).
- **chol**: Serum cholesterol (in mg/dl).
- **fbs**: Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false).
- **restecg**: Resting electrocardiographic results (values from 0 to 2).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise induced angina (1 = yes, 0 = no).
- **oldpeak**: Depression induced by exercise relative to rest.
- **slope**: Slope of the peak exercise ST segment (values from 0 to 2).
- **ca**: Number of major vessels colored by fluoroscopy (values from 0 to 3).
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect).
- **target**: 1 if heart disease is present, 0 if not.

## Steps Involved in the Project

### 1. Data Collection & Preprocessing

The dataset is cleaned and preprocessed for model training. Some of the key steps include:
- **Handling Missing Values**: Any missing or null values in the dataset are handled appropriately.
- **Feature Scaling**: Features like cholesterol, resting blood pressure, and max heart rate are scaled using techniques like normalization or standardization.
- **Encoding Categorical Variables**: Some categorical variables are encoded into numerical formats for model training.

### 2. Exploratory Data Analysis (EDA)

EDA is performed to understand the distribution of data and relationships between features. Some common EDA steps include:
- **Visualizing Distributions**: Using histograms, box plots, and violin plots to understand the distribution of each feature.
- **Correlation Heatmap**: A correlation heatmap is used to visualize the relationships between features and identify any strong correlations.
- **Class Distribution**: The distribution of the target variable (heart disease presence) is explored.

### 3. Model Building

- **Model Selection**: A **Random Forest Classifier** is chosen for its ability to handle both categorical and numerical data and its robustness in producing accurate predictions.
- **Model Training**: The model is trained on the preprocessed data. Hyperparameter tuning is done using techniques like Grid Search or Random Search to find the best-performing model.
  
### 4. Model Evaluation

To evaluate the model's performance, the following metrics are calculated:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of positive predictions that are actually positive.
- **Recall**: The proportion of actual positives that are correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A matrix that shows the true positives, true negatives, false positives, and false negatives.
- **ROC Curve and AUC**: To visualize the model's ability to discriminate between classes.

### 5. Model Deployment

- **Flask Web Application**: A simple Flask app is created to serve the trained model as a REST API. The app allows users to input their health information through a web form and receive predictions on whether they are at risk for heart disease.
- **API Integration**: The model is integrated into the Flask app, where the input data is collected via a form, passed to the model, and a prediction is returned to the user.

### 6. Real-Time Prediction

- **User Input**: The user is asked to fill in the health parameters through the web interface.
- **Model Prediction**: The trained Random Forest model processes the input and returns the prediction.
- **Response**: The user is informed whether they are likely to have heart disease (1) or not (0).

## Technologies Used

- **Python**: Programming language used for model development and deployment.
- **Flask**: Web framework used to create the API for real-time predictions.
- **Pandas & NumPy**: Libraries used for data manipulation and preprocessing.
- **Scikit-learn**: For building the Random Forest Classifier model.
- **Matplotlib & Seaborn**: For visualizing the data during exploratory analysis.
- **Jinja2**: Templating engine used by Flask to render HTML pages.
- **Joblib**: For saving and loading the trained model.

## How to Run the Project Locally

Follow the steps below to run the project on your local machine:

### 1. Clone the Repository
Clone this GitHub repository to your local machine:

`git clone <repository-url>`

### **2.Install Dependencies**
Navigate to the project folder and install the required dependencies:

`cd <project-folder>`
`pip install -r requirements.txt`

### 3. Run the Flask Application
Run the Flask app:

`python app.py`

This will start a local web server. Open your web browser and go to `http://127.0.0.1:5000` to see the app in action.

### 4. Interact with the Model
Once the app is running, you can input the health parameters into the form and submit it to receive predictions.

### Conclusion
This project demonstrates how to build a machine learning model for heart disease prediction and deploy it using Flask. The Random Forest model provides a solid foundation for making predictions based on user inputs. By deploying the model as a web application, it is made accessible to anyone who wants to check their risk for heart disease based on the given parameters.
