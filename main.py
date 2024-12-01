from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import pickle

# Function to make prediction
def predict_y(input_data):
    prediction = models.predict([input_data])
    return prediction[0]

app = FastAPI()
# Load the machine learning model
models = joblib.load('bestModz_RandomForest.pkl')

class InputData(BaseModel):
    age: int
    job: int
    marital: int
    education: int
    default: int
    housing: int
    loan: int
    contact: int 
    duration: float 
    campaign: int 
    pdays: int 
    previous: int
    poutcome: int 

@app.get("/")
def read_root():
       return {"message": "validation testing"}

@app.post('/predict')
def predict(df: InputData):
    input_data = [df.age, df.job, df.marital, df.education, df.default, df.housing, df.loan, df.contact, df.duration, df.campaign, df.pdays, df.previous, df.poutcome]
    prediction = predict_y(input_data)
    return {'prediction': int(prediction)}
