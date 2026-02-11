#how to predict house price based on user input
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
            "area": float(request.POST["area"]),
            "bedrooms": int(request.POST["bedrooms"]),
            "bathrooms": int(request.POST["bathrooms"]),
            "floors": int(request.POST["floors"]),
            "condition":1 if request.POST["condition"] == "Good" else 2 if request.POST["condition"] == "Average" else 3,
            "garage": 1 if request.POST["garage"] == "Yes" else 0,
            "house_age": int(request.POST["house_age"]),
            "yr_built": int(request.POST["yr_built"]),
            "yr_renovated": int(request.POST["yr_renovated"]),
            "sqft_basement": float(request.POST["sqft_basement"]),
            "sqft_lot": float(request.POST["sqft_lot"]),
            "sqft_above": float(request.POST["sqft_above"]),
            "waterfront":1 if request.POST["waterfront"] == "Yes" else 0,
            "grade": int(request.POST["grade"]),
            "view": int(request.POST["view"]),
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

    return render(request, "index.html", {"prediction": prediction})
