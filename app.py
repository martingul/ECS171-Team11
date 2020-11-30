from typing import Optional
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import models
import pickle


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
    # print("Administrative", Administrative)
    # print("Administrative_Duration",Administrative_Duration)
    # print("Informational",Informational)
    # print("Informational_Duration",Informational_Duration)
    # print("ProductRelated",ProductRelated)
    # print("BounceRates",BounceRates)
    # print("ExitRates",ExitRates)
    # print("PageValues",PageValues)
    # print("SpecialDay",SpecialDay)
    # print("Weekend",Weekend)
    # print("Month",Month)
    # print("OperatingSystems",OperatingSystems)
    # print("Browser",Browser)
    # print("Region",Region)
    # print("TrafficType",TrafficType)
    # print("VisitorType",VisitorType)

    # load pickle files and use them directly
    with open("sklean_MLP.pkl", 'rb') as file:  
        Pickled_sklean_MLP= pickle.load(file)

    print(Pickled_sklean_MLP)

    return templates.TemplateResponse(
        "index.html", {"request": request, "Administrative": Administrative}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)