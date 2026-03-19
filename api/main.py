from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from api.schema import userInput
from api.all_class import change_OnehotEncoder, load_model, model_path
import joblib
import pathlib



app = FastAPI()


@app.get('/')
def home():
    return {'message': 'This is the home url of salary_prediction API'}

@app.get('/hello')
def hello():
    return {'message': 'Hello'}


@app.get('/detail')
def detail():
    return {
        "message": "Welcome to the Salary Prediction API for India! 🚀 Send your input to /predict to get salary predictions with detailed insights"
    }

@app.post('/predict')
def predict(data: userInput):
    md_path = model_path()
    model = load_model(md_path)
    vailidate_data = change_OnehotEncoder(data=data)
    pr_valu = model.predict(vailidate_data)
    return JSONResponse(status_code=201, content=pr_valu.tolist())


    

