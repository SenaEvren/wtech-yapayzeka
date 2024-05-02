from fastapi import FastAPI
from pydantic import BaseModel 
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = FastAPI()

# Tahmin için giriş verisi için veri modeli oluşturma
class InputData(BaseModel):
    Type_of_Travellers: int
    Seat_Types: int
    Seat_Comfort: int
    Cabin_Staff_Service: int
    Ground_Service: int
    Food_Beverages: int
    Wifi_Connectivity: int
    Inflight_Entertainment: int
    Value_For_Money: int

# Tahmin yapma endpointi
@app.post("/predict/")
async def predict(input_data: InputData):
    my_model = load_model("model.h5")

    sc = StandardScaler()
    data = pd.DataFrame([input_data.dict()])
    sc.fit(data)
    data = sc.transform(data)
    
    prediction = my_model.predict(data)
    prediction = prediction[0]
    
    advice = prediction.argmax()
    
    if advice == 0:
        advice = "NO"
    elif advice == 1:
        advice = "YES"

    return {"Recommendation": advice}
