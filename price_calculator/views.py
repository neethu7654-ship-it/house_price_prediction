from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import joblib
import pandas as pd
import os

# Load trained model
MODEL_PATH = os.path.join(settings.BASE_DIR.parent, 'ml_model', 'house_price_model.pkl')

model = joblib.load(MODEL_PATH)

def predict_price(request):
    prediction = None

    if request.method == "POST":
        data = {
            "Area": float(request.POST["area"]),
            "Bedrooms": int(request.POST["bedrooms"]),
            "Bathrooms": int(request.POST["bathrooms"]),
            "Floors": int(request.POST["floors"]),
            "Condition": request.POST["condition"],
            "Garage": request.POST["garage"],
            "Location": request.POST["location"],
            "House_Age": int(request.POST["house_age"]),
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

    return render(request, "index.html", {"prediction": prediction})
