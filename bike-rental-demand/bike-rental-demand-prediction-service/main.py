from fastapi import FastAPI

from .routers import model

service=FastAPI()

service.include_router(model.router)

@service.get("/")
async def get_service():
    return {'message': 'Service is running'}