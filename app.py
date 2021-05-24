import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('dtree.pkl', 'rb'))
dataset = pd.read_csv('diabetes.csv')
X = dataset[
    ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
     "Outcome"]]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                DiabetesPedigreeFunction, Age, Outcome):
    output = model.predict(sc.transform(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome]]))
    print("Diabetes =", output)
    if output == [1]:
        prediction = "Diabetes"
    else:
        prediction = "NOT Diabetes"
    print(prediction)
    return prediction


def main():
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("Passenger Will Shave Diabetes or Not using Decision Tree")

    Pregnancies = st.number_input('Insert Pregnancies')
    Glucose = st.number_input('Insert a Glucose', 18, 60)
    BloodPressure = st.number_input('Insert a BloodPressure', 0, 10)
    SkinThickness = st.number_input('Insert a SkinThickness', 1, 10)
    Insulin = st.number_input('Insert a Insulin', 18, 60)
    BMI = st.number_input('Insert a BMI', 30, 60)
    DiabetesPedigreeFunction = st.number_input("Insert DiabetesPedigreeFunction", 1, 15000)
    Age = st.number_input('Insert a Age', 18, 60)


result = ""
if st.button("Predict"):
    result = predict_note_authentication(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)
    st.success('Model has predicted {}'.format(result))
if st.button("About"):
    st.subheader("Developed by Ritik Banger")
    st.subheader("Department of Computer Engineering")

if __name__ == '__main__':
    main()
