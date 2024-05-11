import typing as T
import numpy as np
import torch

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from model_architecture import Net
from get_data import get_doodle_names

# INDSÆT STI TIL GEMT MODEL
SAVED_MODEL_PATH = 'saved_models/2024-05-11-15-04-57cqei.pt'
CLASSES = get_doodle_names()


class Request(BaseModel):
    """Pydantic model der indeholder et billede til at forudsige klassen af"""
    image: T.List[float]
    class Config:
        arbitrary_types_allowed = True

app = FastAPI()

@app.get("/")
async def heartbeat() -> T.Dict[str, str]:
    """Checker om serveren kører"""
    return {"message": "Hello World"}

@app.get("/predict")
def predict(r: Request) -> T.Dict[str, str]:
    """Prædikterer klassen af billedet i requestet"""
    image = np.asarray(r.image).reshape(28,28)
    model = Net(C=len(CLASSES))
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    model.eval()

    y_hat, y_hat_prob = model.predict(image)
    return {"prediction": CLASSES[y_hat], "probability": y_hat_prob}

def main():
    """Starter FastAPI serveren"""
    uvicorn.run(app, host="localhost", port=8000, reload=True, workers=2)

if __name__ == "__main__":
    main()