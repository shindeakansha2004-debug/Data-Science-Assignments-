import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction App")
st.write("Enter passenger details below and check survival prediction")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs same as training
sex = 1 if sex == "male" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_dict[embarked]

# Prepare input
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("🎉 This passenger would have SURVIVED!")
    else:
        st.error("💀 This passenger would NOT have survived.")
