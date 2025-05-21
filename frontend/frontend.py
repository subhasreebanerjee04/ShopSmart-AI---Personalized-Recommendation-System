# frontend.py
import streamlit as st
import requests
import time
from datetime import datetime

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
    
    .debug-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
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

# Check backend connection
def check_backend_connection():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=3)
        return response.status_code == 200
    except Exception as e:
        st.sidebar.error(f"Backend connection failed: {str(e)}")
        return False

if not check_backend_connection():
    st.error("⚠️ Backend service is not available. Please ensure the backend is running.")
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Select recommendation type",
        ["For You", "Similar Items", "Smart Picks", "Product Search"],
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
            # Handle missing fields with defaults
            title = product.get('title', 'No title available')
            brand = product.get('brand', 'Brand not specified')
            price = product.get('price', 0)
            rating = product.get('avg_rating', 0)
            image_url = product.get('image_url', '')
            item_id = product.get('item_id', '')
            
            st.markdown(f"""
            <div class="product-card">
                <div style="text-align:center; margin-bottom:0.5rem;">
                    <img src="{image_url}" style="max-height:180px; width:auto; border-radius:4px;" 
                         onerror="this.src='https://via.placeholder.com/150?text=No+Image'">
                </div>
                <div class="product-title">{title}</div>
                <div class="product-brand">by {brand}</div>
                <div class="product-price">${price:.2f}</div>
                <div class="product-rating">
                    {"⭐" * int(round(rating))} 
                    <span style="color:#555;">({rating:.1f})</span>
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
                <div class="debug-info">ID: {item_id}</div>
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
        [f"user_{i}" for i in range(500)],
        key="user_select"
    )
    
    if st.button("Get My Recommendations", key="cf_button"):
        if not user_id:
            st.warning("Please select a user profile")
        else:
            with st.spinner("Analyzing your preferences..."):
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{BACKEND_URL}/recommend/cf",
                        params={"user_id": user_id.lower(), "n": 6},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        elapsed_time = time.time() - start_time
                        
                        if "recommendations" not in data:
                            st.error("Unexpected response format from server")
                            st.write(data)  # Debug output
                        else:
                            st.success(f"✨ Found {len(data['recommendations'])} personalized recommendations (took {elapsed_time:.1f}s)")
                            display_product_grid(data["recommendations"])
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the recommendation service. Please check your internet connection.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.write("Full error details:", e)
    
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
        [f"item_{i}" for i in range(200)],
        key="item_select"
    )
    
    if st.button("Find Similar Items", key="cb_button"):
        if not item_id:
            st.warning("Please select a product")
        else:
            with st.spinner("Searching our catalog for similar products..."):
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{BACKEND_URL}/recommend/cb",
                        params={"item_id": item_id.lower(), "n": 6},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        elapsed_time = time.time() - start_time
                        
                        if "recommendations" not in data:
                            st.error("Unexpected response format from server")
                            st.write(data)  # Debug output
                        else:
                            # Show the selected product if available
                            selected_product = next(
                                (p for p in data["recommendations"] 
                                if p.get("item_id", "").lower() == item_id.lower()),
                                None
                            )
                            
                            if selected_product:
                                st.markdown("#### You selected:")
                                display_product_grid([selected_product], cols=1)
                            
                            st.success(f"🔍 Found {len(data['recommendations'])} similar products (took {elapsed_time:.1f}s)")
                            display_product_grid(data["recommendations"])
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the recommendation service. Please check your internet connection.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.write("Full error details:", e)
    
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Smart Picks":
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
            [f"user_{i}" for i in range(500)],
            key="hybrid_user"
        )
    with col2:
        item_id = st.selectbox(
            "Select a product you're interested in",
            [f"item_{i}" for i in range(200)],
            key="hybrid_item"
        )
    
    if st.button("Get Smart Recommendations", key="hybrid_button"):
        if not user_id or not item_id:
            st.warning("Please select both a user profile and a product")
        else:
            with st.spinner("Calculating your perfect recommendations..."):
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{BACKEND_URL}/recommend/hybrid",
                        params={
                            "user_id": user_id.lower(),
                            "item_id": item_id.lower(),
                            "n": 6
                        },
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        elapsed_time = time.time() - start_time
                        
                        if "recommendations" not in data:
                            st.error("Unexpected response format from server")
                            st.write(data)  # Debug output
                        else:
                            # Show the selected product if available
                            selected_product = next(
                                (p for p in data["recommendations"] 
                                 if p.get("item_id", "").lower() == item_id.lower()),
                                None
                            )
                            
                            if selected_product:
                                st.markdown("#### Based on:")
                                display_product_grid([selected_product], cols=1)
                            
                            st.success(f"🎯 Found {len(data['recommendations'])} smart picks for you (took {elapsed_time:.1f}s)")
                            display_product_grid(data["recommendations"])
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the recommendation service. Please check your internet connection.")
                except requests.exceptions.Timeout:
                    st.error("The request took too long. Please try again with different parameters.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.write("Full error details:", e)
    
    st.markdown("</div>", unsafe_allow_html=True)

else:  # Product Search
    st.markdown("""
    <div class="recommendation-section">
        <div class="section-title">🔍 Exact Product Search</div>
        <div class="section-subtitle">
            Find a specific product by its exact name
        </div>
    """, unsafe_allow_html=True)
    
    search_query = st.text_input("Enter exact product name", 
                               placeholder="e.g., 'Product 15'",
                               key="exact_product_search")

    if st.button("Search Exact Product", key="exact_search_button"):
        if not search_query.strip():
            st.warning("Please enter a product name to search")
        else:
            with st.spinner(f"Searching for exact match: '{search_query}'..."):
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{BACKEND_URL}/search/product",
                        params={"product_name": search_query},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        elapsed_time = time.time() - start_time
                        
                        if data.get("product") is None:
                            st.error(f"Product not found: '{search_query}'")
                            st.markdown(f"""
                            <div style="text-align:center; padding:2rem; color:#666;">
                                <p style="font-size:1.2rem; margin-bottom:1rem;">No exact match found for:</p>
                                <p style="font-size:1.5rem; font-weight:bold; color:var(--amazon-dark);">"{search_query}"</p>
                                <p style="margin-top:1rem;">Please check the spelling or try a different product name</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            product = data["product"]
                            st.success(f"Found exact match in {elapsed_time:.2f}s")
                            
                            # Display the single product in a detailed view
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(
                                    product.get('image_url', 'https://via.placeholder.com/300?text=No+Image'),
                                    width=300,
                                    caption=product.get('title', 'Product Image')
                                )
                            with col2:
                                st.markdown(f"""
                                <div style="margin-left:1rem;">
                                    <h2 style="color:var(--amazon-dark); margin-bottom:0.5rem;">{product.get('title', 'No Title')}</h2>
                                    <p style="color:#555; margin-bottom:0.5rem;">by {product.get('brand', 'Unknown Brand')}</p>
                                    <div style="display:flex; align-items:center; margin-bottom:1rem;">
                                        <span style="font-size:1.8rem; color:#B12704; font-weight:bold;">
                                            ${product.get('price', 0):.2f}
                                        </span>
                                        <span style="margin-left:2rem; color:#FFA41C; font-weight:bold;">
                                            {"⭐" * int(round(product.get('avg_rating', 0)))} 
                                            <span style="color:#555;">({product.get('avg_rating', 0):.1f})</span>
                                        </span>
                                    </div>
                                    <p style="margin-bottom:0.5rem;"><strong>Category:</strong> {product.get('category', 'N/A')}</p>
                                    <p style="margin-bottom:1.5rem;"><strong>Product ID:</strong> {product.get('item_id', 'N/A')}</p>
                                    <button style="
                                        background:linear-gradient(to bottom, #FFD814, #F7CA00);
                                        color:#111; border:none; padding:0.75rem 1.5rem; 
                                        border-radius:4px; cursor:pointer;
                                        font-weight:600; border:1px solid #F2C200;
                                        box-shadow:0 1px 2px rgba(0,0,0,0.1);
                                        font-size:1.1rem;
                                        margin-right:1rem;
                                    ">
                                        Add to Cart
                                    </button>
                                    <button style="
                                        background:white;
                                        color:#111; border:1px solid #DDD; padding:0.75rem 1.5rem; 
                                        border-radius:4px; cursor:pointer;
                                        font-weight:600;
                                        box-shadow:0 1px 2px rgba(0,0,0,0.1);
                                        font-size:1.1rem;
                                    ">
                                        View Details
                                    </button>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection Error: Could not reach the server. Please check your internet connection.")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request Timeout: The search took too long. Please try again.")
                except Exception as e:
                    st.error(f"⚠️ Unexpected Error: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center; color:#555; margin-top:3rem; padding:1.5rem 0; border-top:1px solid #EEE;">
    <p style="margin:0.5rem 0;">ShopSmart AI Recommendation System</p>
    <p style="margin:0.5rem 0; font-size:0.9rem;">Note: This demo uses synthetic data to simulate recommendations</p>
    <p style="margin:0.5rem 0; font-size:0.8rem; color:#888;">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)
