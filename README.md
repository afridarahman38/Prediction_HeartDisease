# Heart Disease Prediction Using Machine Learning

### Project Overview
This project aims to predict heart disease using machine learning techniques based on various health and lifestyle factors. The dataset contains 319,796 records with features such as BMI, Smoking, Alcohol Consumption, Stroke History, Physical & Mental Health, Age, Diabetes, Physical Activity, Sleep Time, and more.

By leveraging different machine learning models, the system can analyze patient data and predict the likelihood of heart disease.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### Technologies Used
- Machine Learning Models: Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM)
- Metrics Used for Evaluation: Precision, Recall, and F1-Score
- Programming Language: Python
- Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

### Model Comparison & Findings
All models provided similar accuracy; however, after evaluating Precision, Recall, and F1-Score, Random Forest emerged as the best choice due to its highest recall for class 1 while maintaining good precision. This makes it an effective model for identifying potential heart disease cases with fewer false negatives.

### Features in the Dataset
Health Factors: BMI, Physical & Mental Health, Sleep Time
Lifestyle Factors: Smoking, Alcohol Consumption, Physical Activity
Medical History: Stroke, Diabetes, Kidney Disease, Skin Cancer, Asthma
Demographics: Sex, Age Category, Race


### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
