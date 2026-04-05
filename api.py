from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

app = FastAPI()

model = mlflow.sklearn.load_model("served_model")

class CreditInput(BaseModel):
    person_age: int
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float

@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running"}

@app.post("/predict")
def predict(data: CreditInput):
    df = pd.DataFrame([data.model_dump()])

    df = df[
        [
            "person_age",
            "person_income",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
        ]
    ]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability_of_default": float(probability)
    }