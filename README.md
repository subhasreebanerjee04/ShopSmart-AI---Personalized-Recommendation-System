# 🛍️ ShopSmart AI - Personalized Recommendation System

ShopSmart AI is an intelligent recommendation system that combines collaborative filtering and content-based techniques to deliver personalized product suggestions. Designed with e-commerce in mind.

# ✨ Key Features

🎯 Hybrid Recommendations	Combines collaborative filtering + content-based approaches

⚡ FastAPI Backend	RESTful API with 3 recommendation endpoints

💅 Streamlit Frontend	Amazon-inspired UI with interactive widgets

📊 Synthetic Data	Self-contained demo with generated user-item interactions


# 🧩 System Architecture
![deepseek_mermaid_20250520_5b5d14](https://github.com/user-attachments/assets/5c067a77-b37e-4a92-a9b5-1e3863faa694)

# 🚀 Quick Start

1. Clone repository

git clone https://github.com/yourusername/ShopSmart-AI.git

cd ShopSmart-AI

2. Install backend dependencies

cd backend

pip install -r requirements.txt

3. Install frontend dependencies

cd ../frontend

pip install -r requirements.txt

# 🏃Running the System

1. Start backend server (in backend directory)

uvicorn main:app --reload

2. In another terminal, start frontend (in frontend directory)

streamlit run frontend.py


Access the web interface at http://localhost:8501

API documentation at http://localhost:8000/docs

# 📜 License
Distributed under the MIT License. See LICENSE for more information.
