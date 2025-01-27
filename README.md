# Heart Disease Prediction using Random Forest Classifier and Flask API

## Project Overview

This project demonstrates the complete process of developing a data science solution for heart disease prediction, from data collection and preprocessing to model deployment using Flask. The model classifies whether a person has heart disease based on various health attributes. We use the **Random Forest Classifier** for model training and deploy it as a **Flask** web application to allow real-time predictions.<br/>

## Dataset Description <br/>

The dataset includes health-related attributes such as age, sex, cholesterol, blood pressure, chest pain type, and other vital signs to predict the presence of heart disease. The target variable is binary, indicating whether the person has heart disease (1) or not (0).<br/>

## Steps Involved in the Project <br/>

### 1. Data Collection & Preprocessing<br/>
- The dataset is preprocessed to handle missing values, scale features, and encode categorical variables.<br/>
- Exploratory Data Analysis (EDA) is conducted to visualize data distribution and relationships between variables.<br/>

### 2. Model Building<br/>
- The **Random Forest Classifier** is chosen for training the model due to its robustness and accuracy.<br/>
- The model is trained and fine-tuned using the dataset to predict heart disease.<br/>

### 3. Model Evaluation<br/>
- Evaluation metrics such as accuracy, precision, recall, F1-score, and a confusion matrix are used to assess model performance.<br/>

### 4. Model Deployment<br/>
- Flask is used to deploy the model as a web application, where users can input data and get predictions.<br/>
- The Flask API accepts inputs through a web form, processes them, and provides predictions in real time.<br/>

## Technologies Used<br/>
- **Python**: Programming language for model development and deployment.<br/>
- **Pandas & NumPy**: For data preprocessing and manipulation.<br/>
- **Matplotlib & Seaborn**: For data visualization.<br/>
- **Scikit-learn**: For training the Random Forest Classifier model.<br/>
- **Flask**: For deploying the machine learning model as a web application.<br/>


