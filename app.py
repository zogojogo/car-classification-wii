from fastapi import FastAPI, File, UploadFile
from starlette.responses import Response
import uvicorn
from src.load_model import load_model
from src.predict import forward_pass
from models.network import Net
from torch.nn import functional as F
import time

model = load_model('models/checkpoint_74.27')

# Create Fast API
app = FastAPI()

@app.get("/")
async def index():
    return {"messages": "Open the documentations /docs or /redoc"}

@app.post("/predict_car")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        start_time = time.time()
        conf, predicted_id, classes = forward_pass(image, model, mode='api', topk=5)
        end_time = time.time()

        return {
            "filename": str(file.filename),
            "contentype": str(file.content_type),
            "predicted class": str(classes[0]),
            "confidence": str(conf[0]),
            "inference time": str(end_time - start_time)
        }
    except:
        return Response("Internal server error", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
