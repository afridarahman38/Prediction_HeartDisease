import streamlit as st
import numpy as np
import pickle  # To load pre-trained models

# # Load pre-trained models
# with open('heart_disease_RandomForest', 'rb') as model_file:
#     model = pickle.load(model_file)

# Function to make predictions
def predict_heart_disease(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]  
    # Probability of disease
    return prediction, probability

# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Enter your details below to check your heart disease risk.")

# Sidebar for user input
st.sidebar.header("User Input Parameters")
BMI = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
Smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
AlcoholDrinking = st.sidebar.selectbox("Alcohol Drinking", ["No", "Yes"])
Stroke = st.sidebar.selectbox("Stroke History", ["No", "Yes"])
PhysicalHealth = st.sidebar.slider("Physical Health (Days unwell in last month)", 0, 30, 5)
MentalHealth = st.sidebar.slider("Mental Health (Days unwell in last month)", 0, 30, 5)
DiffWalking = st.sidebar.selectbox("Difficulty Walking", ["No", "Yes"])
Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
AgeCategory = st.sidebar.selectbox("Age Category", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], index=2)
Diabetic = st.sidebar.selectbox("Diabetic", ["No", "Yes"])
PhysicalActivity = st.sidebar.selectbox("Physical Activity", ["No", "Yes"])
GenHealth = st.sidebar.selectbox("General Health", ["Excellent", "Good", "Fair", "Poor"])
SleepTime = st.sidebar.slider("Average Sleep Time (hours)", 3, 12, 7)
Asthma = st.sidebar.selectbox("Asthma", ["No", "Yes"])
KidneyDisease = st.sidebar.selectbox("Kidney Disease", ["No", "Yes"])
SkinCancer = st.sidebar.selectbox("Skin Cancer", ["No", "Yes"])

# Convert categorical inputs to numerical values
mapping = {"No": 0, "Yes": 1, "Male": 1, "Female": 0, "Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}
AgeMapping = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65+": 6}

input_features = [
    BMI, mapping[Smoking], mapping[AlcoholDrinking], mapping[Stroke],
    PhysicalHealth, MentalHealth, mapping[DiffWalking], mapping[Sex],
    AgeMapping[AgeCategory], mapping[Diabetic], mapping[PhysicalActivity],
    mapping[GenHealth], SleepTime, mapping[Asthma], mapping[KidneyDisease],
    mapping[SkinCancer]
]

if st.sidebar.button("Predict Heart Disease Risk"):
    prediction, probability = predict_heart_disease(input_features)
    if prediction == 1:
        st.error(f"⚠️ High Risk: {probability*100:.2f}% probability of heart disease.")
    else:
        st.success(f"✅ Low Risk: {probability*100:.2f}% probability of heart disease.")

st.write("---")
st.write("👨‍⚕️ **Disclaimer:** This prediction is based on machine learning and should not replace medical advice.")



# import streamlit as st
# import numpy as np
# import pickle

# # Load trained model
# with open("heart_disease_model.pkl", "rb") as file:
#     model = pickle.load(file)

# # Streamlit App
# st.title("Heart Disease Prediction")
# st.write("Enter patient details to predict the likelihood of heart disease.")

# # User inputs
# age = st.number_input("Age", min_value=20, max_value=100, value=50)
# sex = st.radio("Sex", ["Male", "Female"])
# cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
# trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
# chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
# thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)

# # Convert categorical variables
# sex = 1 if sex == "Male" else 0

# # Prediction
# if st.button("Predict"):
#     input_data = np.array([[age, sex, cp, trestbps, chol, thalach]])
#     prediction = model.predict(input_data)

#     if prediction[0] == 1:
#         st.error("The model predicts a high risk of heart disease. Please consult a doctor.")
#     else:
#         st.success("The model predicts a low risk of heart disease.")

# # Run with: streamlit run app.py
