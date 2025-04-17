from fastapi import APIRouter
from pydantic import BaseModel
from app.chatbot import predict_intent, get_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    intent: str
    confidence: float
    response: str

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    intent, confidence = predict_intent(request.message)
    if confidence > 0.6:
        response = get_response(intent)
    else:
        response = "Mình chưa hiểu rõ ý bạn, bạn có thể nói lại không?"
    
    return ChatResponse(intent=intent, confidence=confidence, response=response)
