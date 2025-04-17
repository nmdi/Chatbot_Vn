import streamlit as st
from src.chatbot_phobert import predict_intent, get_response

# Streamlit setup
st.set_page_config(page_title="PhoBERT ChatBot", page_icon="🤖")
st.title("🤖 ChatBot PhoBERT")
st.write("Chào bạn! Mình là trợ lý ảo dùng mô hình PhoBERT. Cứ nhập nội dung bên dưới nhé!")

# Lưu hội thoại trong session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị hội thoại cũ
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Nhập mới từ người dùng
user_input = st.chat_input("Nhập nội dung...")

if user_input:
    # Hiển thị tin nhắn người dùng
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Dự đoán intent và phản hồi
    intent, confidence = predict_intent(user_input)
    if confidence > 0.6:
        bot_reply = get_response(intent)
    else:
        bot_reply = "Mình chưa hiểu rõ ý bạn, bạn có thể nói lại không?"

    # Hiển thị phản hồi bot
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})