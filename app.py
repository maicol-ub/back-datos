from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configuración CORS para permitir solicitudes desde tu aplicación web (ajusta según sea necesario)
origins = [
    "http://localhost",  # Reemplaza tu_puerto_de_desarrollo con el puerto real
    "https://tu_app_web_en_produccion.com",  # Agrega aquí el dominio de tu aplicación en producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    age: int
    gender: int
    height: int
    weight: int
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    

@app.get("/")
async def root():
    model = load('model_perceptron.joblib')
    model_response = model.predict([[55,1,156,85.0,140,0,0,0,0,0,0]])
    return {"message": model_response.tolist()}

@app.post("/predict")
def predict(item: Item):
    model = load('model_perceptron.joblib')
    data = [
        item.age,
        item.gender,
        item.height,
        item.weight,
        item.ap_hi,
        item.ap_lo,
        item.cholesterol,
        item.gluc,
        item.smoke,
        item.alco,
        item.active
    ]

    model_response = model.predict([data])
    return {"message": model_response.tolist()}
