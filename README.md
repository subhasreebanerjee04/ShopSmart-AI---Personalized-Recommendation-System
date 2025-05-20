#🛍️ ShopSmart AI - Personalized Recommendation Engine
ShopSmart Demo Interactive demo of the recommendation interface

#🌟 Overview
ShopSmart AI is an intelligent recommendation system that combines collaborative filtering and content-based techniques to deliver personalized product suggestions. Designed with e-commerce in mind.

#✨ Key Features

🎯 Hybrid Recommendations	Combines collaborative filtering + content-based approaches
⚡ FastAPI Backend	RESTful API with 3 recommendation endpoints

💅 Streamlit Frontend	Amazon-inspired UI with interactive widgets

📊 Synthetic Data	Self-contained demo with generated user-item interactions

#🚀 Quick Start

 Clone repository

git clone https://github.com/yourusername/ShopSmart-AI.git

cd ShopSmart-AI

 Install backend dependencies

cd backend

pip install -r requirements.txt

 Install frontend dependencies

cd ../frontend

pip install -r requirements.txt

# 🏃Running the System

 Start backend server (in backend directory)

uvicorn main:app --reload

 In another terminal, start frontend (in frontend directory)

streamlit run frontend.py

Access the web interface at http://localhost:8501

API documentation at http://localhost:8000/docs

# 🧩 System Architecture
![deepseek_mermaid_20250520_5b5d14](https://github.com/user-attachments/assets/5c067a77-b37e-4a92-a9b5-1e3863faa694)

# 🛠️ Development

Project Structure

├── backend/
│   ├── main.py               # FastAPI application
│   ├── requirements.txt      # Python dependencies
│   └── models/               # Serialized models
├── frontend/
│   ├── frontend.py           # Streamlit application
│   ├── requirements.txt      # Frontend dependencies
│   └── assets/               # CSS/images
├── docs/                     # Documentation
└── README.md                 # Project overview


# 📜 License
Distributed under the MIT License. See LICENSE for more information.
