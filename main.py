from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from app.api import router

app = FastAPI()
app.include_router(router)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>ChatBot API</title>
        </head>
        <body>
            <h1>Chào mừng đến với ChatBot PhoBERT API!</h1>
            <p>Gửi POST request đến <code>/chat</code> với {"message": "nội dung"} để nhận phản hồi.</p>
        </body>
    </html>
    """

@app.get("/favicon.ico")
async def favicon():
    return Response(content="", media_type="image/x-icon")
