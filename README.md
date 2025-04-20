# Chatbot
Project Description:
Developed an intelligent Vietnamese language chatbot using PhoBERT (a transformer-based model for Vietnamese NLP), integrated with FastAPI for web deployment. The chatbot is designed to understand and respond to user queries based on intents such as greetings, goodbyes, and weather inquiries. The model is trained on custom intent data and is capable of providing dynamic, context-aware responses.

Technologies used:

NLP: PhoBERT, ViTokenizer

Deep Learning: PyTorch

Web Development: FastAPI

Data: JSON, pandas, scikit-learn

API Design: RESTful API, POST method for chat interactions

Model Deployment: Model fine-tuning, optimization with AdamW

UI: HTML, CSS, JavaScript (for web-based chat interface)

Key Contributions:

Implemented a PhoBERT-based text classification model to predict user intents with a high accuracy.

Designed and developed a FastAPI server to handle user queries and generate responses in real-time.

Trained the chatbot model using labeled datasets (patterns and responses) and fine-tuned the PhoBERT model for Vietnamese text.

Created a label encoder to map predicted intent labels to actual response categories.

Developed a user-friendly web interface for chatbot interaction.

Successfully integrated the model and deployed it as a web application to handle live interactions.

Result:

The model achieved high accuracy in classifying intents from user input, providing fast, contextually relevant responses.

The chatbot can handle a variety of common queries like greetings, farewells, weather, and more, improving user engagement and interaction.
