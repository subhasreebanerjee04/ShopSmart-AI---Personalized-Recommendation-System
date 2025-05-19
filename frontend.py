# frontend.py
import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

# Configuration
BACKEND_URL = "http://localhost:8000"

# Page setup
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
    
    .section-title {
        color: var(--amazon-dark);
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        color: var(--amazon-light);
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(to bottom, #F7DCA0, #F0C14B);
        color: #111;
        border-radius: 4px;
        border: 1px solid #A88734;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background: linear-gradient(to bottom, #F5D78E, #EEB933);
        transform: translateY(-1px);
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
    
    .product-title {
        color: #0066C0;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.8rem 0 0.4rem;
        line-height: 1.3;
        min-height: 3rem;
    }
    
    .product-brand {
        color: #555;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .product-price {
        color: #B12704;
        font-weight: 700;
        font-size: 1.3rem;
        margin: 0.5rem 0;
    }
    
    .product-rating {
        color: #FFA41C;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: white;
        border-right: 1px solid #DDD;
    }
    
    .stSelectbox>div>div>select {
        background-color: white;
        border-radius: 4px;
        border: 1px solid #DDD;
    }
    
    .recommendation-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #EEE;
    }
    
    .spinner-container {
        display: flex;
        justify-content: center;
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown("""
<div class="amazon-header">
    <div style="max-width:1200px; margin:0 auto; display:flex; align-items:center; padding:0 2rem;">
        <div style="flex:1;">
            <h1 style="margin:0; font-size:1.8rem; font-weight:700;">ShopSmart AI</h1>
            <p style="margin:0; font-size:1rem; opacity:0.9;">Personalized Recommendations</p>
        </div>
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
    st.markdown("""
    <div style="color:#555;">
        <h4>About This System</h4>
        <p>Our AI analyzes your preferences and product details to deliver personalized recommendations.</p>
        <p>Try different recommendation types to discover products you'll love.</p>
    </div>
    """, unsafe_allow_html=True)

# Helper function to display product grid
def display_product_grid(products, cols=3):
    if not products:
        st.warning("No products found matching your criteria")
        return
    
    columns = st.columns(cols)
    for idx, product in enumerate(products):
        with columns[idx % cols]:
            st.markdown(f"""
            <div class="product-card">
                <div style="text-align:center; margin-bottom:0.5rem;">
                    <img src="{product['image_url']}" style="max-height:180px; width:auto; border-radius:4px;">
                </div>
                <div class="product-title">{product['title']}</div>
                <div class="product-brand">by {product['brand']}</div>
                <div class="product-price">${product['price']:.2f}</div>
                <div class="product-rating">
                    {"⭐" * int(round(product['avg_rating']))} 
                    <span style="color:#555;">({product['avg_rating']:.1f})</span>
                </div>
                <button style="
                    background:linear-gradient(to bottom, #FFD814, #F7CA00);
                    color:#111; border:none; padding:0.5rem; 
                    border-radius:4px; cursor:pointer; width:100%;
                    font-weight:600; border:1px solid #F2C200;
                    box-shadow:0 1px 2px rgba(0,0,0,0.1);
                ">
                    Add to Cart
                </button>
            </div>
            """, unsafe_allow_html=True)

# Recommendation pages
if page == "For You":
    st.markdown("""
    <div class="recommendation-section">
        <div class="section-title">🧑 Personalized For You</div>
        <div class="section-subtitle">
            Discover products tailored to your unique tastes and purchase history
        </div>
    """, unsafe_allow_html=True)
    
    user_id = st.selectbox(
        "Select your user profile",
        [f"User_{i}" for i in range(500)],
        key="user_select"
    )
    
    if st.button("Get My Recommendations", key="cf_button"):
        with st.spinner("Analyzing your preferences..."):
            try:
                response = requests.get(
                    f"{BACKEND_URL}/recommend/cf",
                    params={"user_id": user_id.lower(), "n": 6}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✨ Found {len(data['recommendations'])} personalized recommendations")
                    display_product_grid(data["recommendations"])
                else:
                    st.error("We couldn't fetch recommendations right now. Please try again later.")
            except Exception as e:
                st.error(f"Connection error: Please check your internet connection")
    
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Similar Items":
    st.markdown("""
    <div class="recommendation-section">
        <div class="section-title">🔍 Similar Products</div>
        <div class="section-subtitle">
            Find items similar to products you love
        </div>
    """, unsafe_allow_html=True)
    
    item_id = st.selectbox(
        "Select a product you like",
        [f"Item_{i}" for i in range(200)],
        key="item_select"
    )
    
    if st.button("Find Similar Items", key="cb_button"):
        with st.spinner("Searching our catalog for similar products..."):
            try:
                response = requests.get(
                    f"{BACKEND_URL}/recommend/cb",
                    params={"item_id": item_id.lower(), "n": 6}
                )
                if response.status_code == 200:
                    data = response.json()
                    
                    # Show the selected product
                    selected_product = next((p for p in data["recommendations"] 
                                          if p["item_id"] == item_id.lower()), None)
                    if selected_product:
                        st.markdown("#### You selected:")
                        display_product_grid([selected_product], cols=1)
                    
                    st.success(f"🔍 Found {len(data['recommendations'])} similar products")
                    display_product_grid(data["recommendations"])
                else:
                    st.error("We couldn't find similar items right now. Please try another product.")
            except Exception as e:
                st.error(f"Connection error: Please check your internet connection")
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="recommendation-section">
        <div class="section-title">🎯 Smart Picks</div>
        <div class="section-subtitle">
            Recommendations combining your profile with specific product interests
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.selectbox(
            "Select your profile",
            [f"User_{i}" for i in range(500)],
            key="hybrid_user"
        )
    with col2:
        item_id = st.selectbox(
            "Select a product you're interested in",
            [f"Item_{i}" for i in range(200)],
            key="hybrid_item"
        )
    
    if st.button("Get Smart Recommendations", key="hybrid_button"):
        with st.spinner("Calculating your perfect recommendations..."):
            try:
                response = requests.get(
                    f"{BACKEND_URL}/recommend/hybrid",
                    params={
                        "user_id": user_id.lower(),
                        "item_id": item_id.lower(),
                        "n": 6
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    
                    # Show the selected product
                    selected_product = next((p for p in data["recommendations"] 
                                          if p["item_id"] == item_id.lower()), None)
                    if selected_product:
                        st.markdown("#### Based on:")
                        display_product_grid([selected_product], cols=1)
                    
                    st.success(f"🎯 Found {len(data['recommendations'])} smart picks for you")
                    display_product_grid(data["recommendations"])
                else:
                    st.error("We couldn't generate recommendations right now. Please try again.")
            except Exception as e:
                st.error(f"Connection error: Please check your internet connection")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; color:#555; margin-top:3rem; padding:1.5rem 0; border-top:1px solid #EEE;">
    <p style="margin:0.5rem 0;">ShopSmart AI Recommendation System</p>
    <p style="margin:0.5rem 0; font-size:0.9rem;">Note: This demo uses synthetic data to simulate recommendations</p>
</div>
""", unsafe_allow_html=True)