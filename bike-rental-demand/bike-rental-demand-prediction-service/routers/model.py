from fastapi import APIRouter
from pydantic import BaseModel

from ..model.prediction_model import PredictionModel

router=APIRouter()

model=PredictionModel()

class ForecastRequest(BaseModel):
    periods: int
    temperature: float
    humidity: int
    visibility: int
    functioning_day: bool

@router.post("/service/model/train")
async def train_model():
    await model.train_model()
    return {"message":"Model has been trained"}

@router.post("/service/model/forecast")
async def get_forecast(request: ForecastRequest):
    periods=request.periods
    input={'temp':request.temperature, 'hum':request.humidity, 'vis':request.visibility, 'func':request.functioning_day}
    result=await model.forecast(periods,input)

    response={'forecasted_values':result}
    return response
