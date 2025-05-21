import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os
import warnings
import time
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import nest_asyncio
from typing import Optional, List

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. NLTK Resource Setup
def download_nltk_resources():
    """Ensure all required NLTK resources are available"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()

# Set random seed
np.random.seed(42)

# 2. Data Generation
def generate_synthetic_data(n_users=500, n_items=200):
    """Generate synthetic Amazon-style data"""
    user_ids = [f"user_{i}" for i in range(n_users)]
    item_ids = [f"item_{i}" for i in range(n_items)]
    
    # Generate interactions
    interactions = pd.DataFrame({
        'user_id': np.random.choice(user_ids, 5000),
        'item_id': np.random.choice(item_ids, 5000),
        'rating': np.random.choice([1, 2, 3, 4, 5], 5000, p=[0.05, 0.1, 0.2, 0.3, 0.35])
    })
    
    # Generate items
    categories = ['Electronics', 'Books', 'Home', 'Clothing', 'Sports']
    brands = ['Amazon', 'Apple', 'Samsung', 'Sony', 'Nike']
    
    items = pd.DataFrame({
        'item_id': item_ids,
        'title': [f"Product {i}" for i in range(n_items)],
        'category': np.random.choice(categories, n_items),
        'brand': np.random.choice(brands, n_items),
        'price': np.round(np.random.uniform(10, 500, n_items), 2),
        'avg_rating': np.round(np.random.uniform(3, 5, n_items), 1),
        'image_url': [f"https://picsum.photos/200/300?random={i}" for i in range(n_items)]
    })
    
    return interactions, items

# Generate data
interactions_df, items_df = generate_synthetic_data()

# 3. Data Preparation
train_df, test_df = train_test_split(
    interactions_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index(),
    test_size=0.2,
    random_state=42
)

# 4. Text Processing
def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

items_df['content_features'] = (
    items_df['title'] + ' ' + 
    items_df['brand'] + ' ' + 
    items_df['category']
)
items_df['processed_content'] = items_df['content_features'].apply(preprocess_text)

# 5. Collaborative Filtering
def matrix_factorization(ratings, n_factors=10):
    """Matrix factorization with SVD"""
    matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    if matrix.empty:
        return pd.DataFrame()
    
    U, sigma, Vt = svds(matrix.values, k=min(n_factors, min(matrix.shape)-1))
    sigma = np.diag(sigma)
    return pd.DataFrame(
        np.dot(np.dot(U, sigma), Vt),
        index=matrix.index,
        columns=matrix.columns
    )

cf_predictions = matrix_factorization(train_df)

# 6. Content-Based Filtering
tfidf = TfidfVectorizer(max_features=500)
try:
    tfidf_matrix = tfidf.fit_transform(items_df['processed_content'])
    item_similarity = cosine_similarity(tfidf_matrix)
    item_indices = pd.Series(items_df.index, index=items_df['item_id'])
except Exception as e:
    print(f"TF-IDF error: {e}")
    item_similarity = np.eye(len(items_df))
    item_indices = pd.Series(items_df.index, index=items_df['item_id'])

# 7. Recommender Class
class AmazonRecommender:
    def __init__(self, cf_model, item_similarity, items_df, item_indices, train_df):
        self.cf_model = cf_model
        self.item_similarity = item_similarity
        self.items_df = items_df
        self.item_indices = item_indices
        self.train_df = train_df

    def _cf_recommend(self, user_id, n):
        """Collaborative filtering recommendations"""
        if user_id not in self.cf_model.index:
            return pd.DataFrame()
        
        preds = self.cf_model.loc[user_id].sort_values(ascending=False)
        seen_items = set(self.train_df[self.train_df['user_id'] == user_id]['item_id'])
        recommended_items = preds[~preds.index.isin(seen_items)].index[:n]
        return self.items_df[self.items_df['item_id'].isin(recommended_items)]

    def _cb_recommend(self, item_id, n):
        """Content-based recommendations"""
        if item_id not in self.item_indices.index:
            return pd.DataFrame()
        
        idx = self.item_indices[item_id]
        sim_scores = list(enumerate(self.item_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        similar_indices = [i[0] for i in sim_scores]
        return self.items_df.iloc[similar_indices]

    def _hybrid_recommend(self, user_id, item_id, n):
        """Hybrid recommendations"""
        cf_recs = self._cf_recommend(user_id, n*2)
        cb_recs = self._cb_recommend(item_id, n*2)
        
        if cf_recs.empty:
            return cb_recs.head(n)
        if cb_recs.empty:
            return cf_recs.head(n)
            
        combined = pd.concat([cf_recs, cb_recs]).drop_duplicates()
        return combined.head(n)
    
    def search_products(self, query, fields=None, limit=10):
        """
        Search for products by query string
        
        Args:
            query (str): The search query
            fields (list): List of fields to search in. Default is title, category, brand
            limit (int): Maximum number of results to return
            
        Returns:
            DataFrame: Matching products
        """
        if not query or query.strip() == "":
            return pd.DataFrame()
            
        if fields is None:
            fields = ["title", "category", "brand"]
            
        # Convert query to lowercase for case-insensitive matching
        query = query.lower()
        query_terms = query.split()
        
        # Function to check if any query term is in the text
        def match_any_term(text, terms):
            if pd.isna(text):
                return False
            text = str(text).lower()
            return any(term in text for term in terms)
        
        # Create filters for each field
        filters = []
        for field in fields:
            if field in self.items_df.columns:
                field_filter = self.items_df[field].apply(
                    lambda x: match_any_term(x, query_terms)
                )
                filters.append(field_filter)
        
        # Combine all filters with OR logic
        if not filters:
            return pd.DataFrame()
            
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter | f
            
        # Apply filter and return results
        results = self.items_df[combined_filter].head(limit)
        return results

# Initialize recommender
recommender = AmazonRecommender(
    cf_predictions,
    item_similarity,
    items_df,
    item_indices,
    train_df
)

# 8. FastAPI Application
app = FastAPI(
    title="Amazon-Style Recommender API",
    description="Generates product recommendations and provides search functionality",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend/cf", response_class=JSONResponse)
async def collaborative_filtering(
    user_id: str = Query(..., description="User ID"),
    n: int = Query(5, description="Number of recommendations", ge=1, le=20)
):
    """Collaborative filtering endpoint"""
    try:
        recs = recommender._cf_recommend(user_id, n)
        if recs.empty:
            return JSONResponse(
                content={"message": "No recommendations found", "recommendations": []},
                status_code=200
            )
        return {
            "user_id": user_id,
            "recommendations": recs.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/recommend/cb", response_class=JSONResponse)
async def content_based(
    item_id: str = Query(..., description="Item ID"),
    n: int = Query(5, description="Number of recommendations", ge=1, le=20)
):
    """Content-based endpoint"""
    try:
        recs = recommender._cb_recommend(item_id, n)
        if recs.empty:
            return JSONResponse(
                content={"message": "No similar items found", "recommendations": []},
                status_code=200
            )
        return {
            "item_id": item_id,
            "recommendations": recs.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/recommend/hybrid", response_class=JSONResponse)
async def hybrid(
    user_id: str = Query(..., description="User ID"),
    item_id: str = Query(..., description="Item ID"),
    n: int = Query(5, description="Number of recommendations", ge=1, le=20)
):
    """Hybrid endpoint"""
    try:
        recs = recommender._hybrid_recommend(user_id, item_id, n)
        if recs.empty:
            return JSONResponse(
                content={"message": "No recommendations found", "recommendations": []},
                status_code=200
            )
        return {
            "user_id": user_id,
            "item_id": item_id,
            "recommendations": recs.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/search", response_class=JSONResponse)
async def search(
    query: str = Query(..., description="Search query"),
    fields: Optional[List[str]] = Query(
        None, 
        description="Fields to search in (title, category, brand)",
        example=["title", "brand"]
    ),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=100)
):
    """Search products by query string"""
    try:
        results = recommender.search_products(query, fields, limit)
        if results.empty:
            return JSONResponse(
                content={"message": "No products found", "products": []},
                status_code=200
            )
        return {
            "query": query,
            "fields": fields if fields else ["title", "category", "brand"],
            "count": len(results),
            "products": results.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error searching products: {str(e)}"
        )

@app.get("/search/product", response_class=JSONResponse)
async def search_exact_product(
    product_name: str = Query(..., description="Exact product name to search for")
):
    """Search for a specific product by exact name"""
    try:
        # Case-insensitive exact match
        results = items_df[items_df['title'].str.lower() == product_name.lower()]
        
        if results.empty:
            return JSONResponse(
                content={"message": f"Product '{product_name}' not found", "product": None},
                status_code=200
            )
        
         


        # Return the first (and should be only) match
        return {
            "product_name": product_name,
            "product": results.iloc[0].to_dict()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error searching for product: {str(e)}"
        )

@app.get("/", response_class=JSONResponse)
async def root():
    return {
        "message": "Amazon-Style Recommender System API",
        "endpoints": {
            "/recommend/cf": {
                "description": "Collaborative filtering recommendations",
                "parameters": {
                    "user_id": "string (required)",
                    "n": "integer (optional, default=5)"
                }
            },
            "/recommend/cb": {
                "description": "Content-based recommendations",
                "parameters": {
                    "item_id": "string (required)",
                    "n": "integer (optional, default=5)"
                }
            },
            "/recommend/hybrid": {
                "description": "Hybrid recommendations",
                "parameters": {
                    "user_id": "string (required)",
                    "item_id": "string (required)",
                    "n": "integer (optional, default=5)"
                }
            },
            "/search": {
                "description": "Search products by keyword",
                "parameters": {
                    "query": "string (required)",
                    "fields": "list of strings (optional)",
                    "limit": "integer (optional, default=10)"
                }
            },
            "/search/product": {
                "description": "Search for exact product by name",
                "parameters": {
                    "product_name": "string (required)"
                }
            }
        }
    }

# 9. Run Application
if __name__ == "__main__":
    nest_asyncio.apply()
    print("\nAmazon-Style Recommender System")
    print("-------------------------------")
    print("Starting server at http://localhost:8000")
    print("\nAvailable endpoints:")
    print("GET /recommend/cf?user_id=<user_id>")
    print("GET /recommend/cb?item_id=<item_id>")
    print("GET /recommend/hybrid?user_id=<user_id>&item_id=<item_id>")
    print("GET /search?query=<search_term>")
    print("GET /search/product?product_name=<exact_product_name>")
    print("\nExample valid IDs:")
    print(f"User IDs: user_0 to user_{len(interactions_df['user_id'].unique())-1}")
    print(f"Item IDs: item_0 to item_{len(items_df)-1}")
    print("\nPress CTRL+C to stop the server")
    
    # Save models
    os.makedirs("models", exist_ok=True)
    with open("models/recommender.pkl", "wb") as f:
        pickle.dump(recommender, f)
    items_df.to_csv("models/items.csv", index=False)
    print("\nModels saved to 'models' directory")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
