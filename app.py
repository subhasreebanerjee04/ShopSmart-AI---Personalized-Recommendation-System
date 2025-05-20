# app.py - Main entry point for Streamlit frontend
import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO
import os
from datetime import datetime

# ======================================================================
# Configuration
# ======================================================================

# Backend URL (change this to your deployed backend URL)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")  # Default to localhost

# ======================================================================
# Helper Functions
# ======================================================================

def call_backend(endpoint: str, params: dict):
    """Generic function to call backend API with error handling"""
    try:
        response = requests.get(
            f"{BACKEND_URL}{endpoint}",
            params=params,
            timeout=10  # 10 second timeout
        )
        response.raise_for_status()  # Raise HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend service error: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Invalid response from backend: {str(e)}")
        return None

# ======================================================================
# Streamlit App Configuration
# ======================================================================

st.set_page_config(
    page_title="ShopSmart AI Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Amazon-inspired CSS
st.markdown("""
<style>
    :root {
        --amazon-orange: #FF9900;
        --amazon-dark: #131921;
        --amazon-light: #232F3E;
        --amazon-gray: #EAEDED;
        --amazon-blue: #146EB4;
    }
    
    .stApp {
        background-color: var(--amazon-gray);
        font-family: 'Amazon Ember', Arial, sans-serif;
    }
    
    .amazon-header {
        background: linear-gradient(to bottom, #1a3e6c, #131921);
        color: white;
        padding: 1rem 0;
        margin-bottom: 2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .product-card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
        border: 1px solid #DDD;
    }
    
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        border-color: var(--amazon-orange);
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# UI Components
# ======================================================================

def display_product_grid(products, cols=3):
    """Display products in a responsive grid"""
    if not products:
        st.warning("No products found")
        return
    
    columns = st.columns(cols)
    for idx, product in enumerate(products):
        with columns[idx % cols]:
            st.markdown(f"""
            <div class="product-card">
                <div style="text-align:center; margin-bottom:0.5rem;">
                    <img src="{product['image_url']}" style="max-height:180px; width:auto; border-radius:4px;">
                </div>
                <div style="color:#0066C0; font-weight:600; font-size:1.1rem; margin:0.8rem 0 0.4rem;">
                    {product['title']}
                </div>
                <div style="color:#555; font-size:0.9rem;">
                    by {product['brand']}
                </div>
                <div style="color:#B12704; font-weight:700; font-size:1.3rem;">
                    ${product['price']:.2f}
                </div>
                <div style="color:#FFA41C; font-weight:700;">
                    {"⭐" * int(round(product['avg_rating']))} 
                    <span style="color:#555;">({product['avg_rating']:.1f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ======================================================================
# Page Layout
# ======================================================================

def main():
    """Main application layout"""
    
    # Header
    st.markdown("""
    <div class="amazon-header">
        <div style="max-width:1200px; margin:0 auto; padding:0 2rem;">
            <h1 style="margin:0; font-size:1.8rem; font-weight:700;">ShopSmart AI</h1>
            <p style="margin:0; font-size:1rem; opacity:0.9;">Personalized Recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Select recommendation type",
            ["For You", "Similar Items", "Smart Picks"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown(f"""
        <div style="color:#555;">
            <p>Backend Status: <span style="color:{'green' if BACKEND_URL != 'http://localhost:8000' else 'orange'}">
                {'Production' if BACKEND_URL != 'http://localhost:8000' else 'Local'}
            </span></p>
            <p>Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation pages
    if page == "For You":
        st.markdown("## 🧑 Personalized For You")
        user_id = st.selectbox(
            "Select your user profile",
            [f"user_{i}" for i in range(500)],
            key="user_select"
        )
        
        if st.button("Get My Recommendations"):
            with st.spinner("Analyzing your preferences..."):
                data = call_backend(
                    "/recommend/cf",
                    {"user_id": user_id, "n": 6}
                )
                if data:
                    display_product_grid(data["recommendations"])
    
    elif page == "Similar Items":
        st.markdown("## 🔍 Similar Products")
        item_id = st.selectbox(
            "Select a product you like",
            [f"item_{i}" for i in range(200)],
            key="item_select"
        )
        
        if st.button("Find Similar Items"):
            with st.spinner("Searching our catalog..."):
                data = call_backend(
                    "/recommend/cb",
                    {"item_id": item_id, "n": 6}
                )
                if data:
                    display_product_grid(data["recommendations"])
    
    else:
        st.markdown("## 🎯 Smart Picks")
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.selectbox(
                "Select your profile",
                [f"user_{i}" for i in range(500)],
                key="hybrid_user"
            )
        with col2:
            item_id = st.selectbox(
                "Select a product you're interested in",
                [f"item_{i}" for i in range(200)],
                key="hybrid_item"
            )
        
        if st.button("Get Smart Recommendations"):
            with st.spinner("Calculating your perfect matches..."):
                data = call_backend(
                    "/recommend/hybrid",
                    {"user_id": user_id, "item_id": item_id, "n": 6}
                )
                if data:
                    display_product_grid(data["recommendations"])

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#555; margin-top:2rem;">
        <p>ShopSmart AI Recommendation System</p>
        <p style="font-size:0.8rem;">Note: Demo uses synthetic data | Backend: {BACKEND_URL}</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================================
# Main Execution
# ======================================================================

if __name__ == "__main__":
    main()