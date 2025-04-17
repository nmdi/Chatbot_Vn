from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router

app = FastAPI()

# Gắn API
app.include_router(router)

# Gắn thư mục static (nơi chứa HTML)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Route mặc định: tải giao diện chat
@app.get("/")
def root():
    return FileResponse("app/static/index.html")
