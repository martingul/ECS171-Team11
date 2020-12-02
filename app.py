from typing import Optional
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import models
import pickle
import uvicorn
import joblib
from preprocessing import remove_correlated_features, normalize_vars_with_scaler

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(
    request: Request,
    Administrative: float = Form(...),
    Administrative_Duration: float = Form(...),
    Informational: float = Form(...),
    Informational_Duration: float = Form(...),
    ProductRelated: float = Form(...),
    ProductRelated_Duration: float = Form(...),
    BounceRates: float = Form(...),
    ExitRates: float = Form(...),
    PageValues: float = Form(...),
    SpecialDay: float = Form(...),
    Weekend: int = Form(...),
    Month: str = Form(...),
    OperatingSystems: str = Form(...),
    Browser: str = Form(...),
    Region: str = Form(...),
    TrafficType: str = Form(...),
    VisitorType: str = Form(...),
):
    # load pickle files and use them directly
    with open("sklean_MLP.pkl", "rb") as file:
        Pickled_sklean_MLP = pickle.load(file)

    keras_model = keras.models.load_model("keras")

    with open("svm.pkl", "rb") as file:
        Pickled_svm = pickle.load(file)

    with open("min_max.pkl", "rb") as file:
        scaler = joblib.load('min_max.pkl')

    # create dataframe
    df = pd.DataFrame(
        columns=[
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
            "Weekend",
            "Month_Aug",
            "Month_Dec",
            "Month_Feb",
            "Month_Jul",
            "Month_June",
            "Month_Mar",
            "Month_May",
            "Month_Nov",
            "Month_Oct",
            "Month_Sep",
            "OperatingSystems_1",
            "OperatingSystems_2",
            "OperatingSystems_3",
            "OperatingSystems_4",
            "OperatingSystems_5",
            "OperatingSystems_6",
            "OperatingSystems_7",
            "OperatingSystems_8",
            "Browser_1",
            "Browser_2",
            "Browser_3",
            "Browser_4",
            "Browser_5",
            "Browser_6",
            "Browser_7",
            "Browser_8",
            "Browser_9",
            "Browser_10",
            "Browser_11",
            "Browser_12",
            "Browser_13",
            "Region_1",
            "Region_2",
            "Region_3",
            "Region_4",
            "Region_5",
            "Region_6",
            "Region_7",
            "Region_8",
            "Region_9",
            "TrafficType_1",
            "TrafficType_2",
            "TrafficType_3",
            "TrafficType_4",
            "TrafficType_5",
            "TrafficType_6",
            "TrafficType_7",
            "TrafficType_8",
            "TrafficType_9",
            "TrafficType_10",
            "TrafficType_11",
            "TrafficType_12",
            "TrafficType_13",
            "TrafficType_14",
            "TrafficType_15",
            "TrafficType_16",
            "TrafficType_17",
            "TrafficType_18",
            "TrafficType_19",
            "TrafficType_20",
            "VisitorType_New_Visitor",
            "VisitorType_Other",
            "VisitorType_Returning_Visitor",
        ]
    )

    df.loc[0, :] = 0

    # float
    df.loc[0, "Administrative"] = Administrative
    df.loc[0, "Administrative_Duration"] = Administrative_Duration
    df.loc[0, "Informational"] = Informational
    df.loc[0, "Informational_Duration"] = Informational_Duration
    df.loc[0, "ProductRelated"] = ProductRelated
    df.loc[0, "ProductRelated_Duration"] = ProductRelated_Duration
    df.loc[0, "BounceRates"] = BounceRates
    df.loc[0, "ExitRates"] = ExitRates
    df.loc[0, "PageValues"] = PageValues
    df.loc[0, "SpecialDay"] = SpecialDay

    # int
    df.loc[0, "Weekend"] = Weekend

    # str
    df.loc[0, Month] = 1
    df.loc[0, OperatingSystems] = 1
    df.loc[0, Browser] = 1
    df.loc[0, Region] = 1
    df.loc[0, TrafficType] = 1
    df.loc[0, VisitorType] = 1

    df = normalize_vars_with_scaler(df, scaler)

    # Remove highly correlated features behind the scenes
    df = remove_correlated_features(df)

    # sk_mlp model
    predicted_sk_mlp = Pickled_sklean_MLP.predict(df)
    print("sk_mlp's prediction: ", predicted_sk_mlp)

    # keras model
    X = np.asarray(df).astype(np.float32)
    predicted_keras = keras_model.predict(X)
    if predicted_keras > 0.5:
        predicted_keras = 1
    else:
        predicted_keras = 0
    predicted_keras = [predicted_keras]
    print("keras's prediction: ", predicted_keras)

    # svm model
    predicted_svm = Pickled_svm.predict(df)
    print("svm's prediction: ", predicted_svm)

    return templates.TemplateResponse(
        "display.html",
        {
            "request": request,
            "predicted_sk_mlp": predicted_sk_mlp,
            "predicted_keras": predicted_keras,
            "predicted_svm": predicted_svm,
        },
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
