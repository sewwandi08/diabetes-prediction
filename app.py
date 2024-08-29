import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

class DiabetesData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


app = FastAPI()

# Load the pre-trained model from joblib file
classifier = joblib.load("./diabets-prediction.joblib")


@app.get('/')
def index():
    return {'message': 'Diabetes Prediction ML API'}


@app.post('/predict')
def predict_diabetes(data: DiabetesData):
    data = data.dict()
    input_data = [
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]

    # Use the input data for prediction
    prediction = classifier.predict([input_data])

    # Interpret the prediction result
    if prediction[0] == 0:
        result = 'The person is not diabetic'
    else:
        result = 'The person is diabetic'

    return {'prediction': result}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

