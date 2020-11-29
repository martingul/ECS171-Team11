from typing import Optional
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount('/static', StaticFiles(directory='static'), name='static')



@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post("/predict/")
async def predict(request: Request,Administrative:int = Form(...)):
    print('Administrative',Administrative)
    return templates.TemplateResponse('index.html', {'request': request,'Administrative':Administrative})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)