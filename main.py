from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="PhoBERT Chatbot API",
    description="FastAPI phục vụ chatbot sử dụng mô hình PhoBERT",
    version="1.0.0"
)

app.include_router(router)
