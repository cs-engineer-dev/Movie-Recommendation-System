import os
from pathlib import Path
from io import BytesIO
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD
import requests
from dotenv import load_dotenv


load_dotenv()  
OMDB_API_KEY = os.getenv("OMDB_API_KEY") or ""
TMDB_API_KEY = os.getenv("TMDB_API_KEY") or ""
try:
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))
except ValueError:
    HYBRID_ALPHA = 0.7

# Paths 
DATA_DIR = "./data/processed"
MODELS_DIR = "./models"
ASSETS_DIR = "./assets"
POSTER_DIR = Path(ASSETS_DIR) / "posters"
PLACEHOLDER_PATH = Path(ASSETS_DIR) / "placeholder.png"
os.makedirs(POSTER_DIR, exist_ok=True)

# Load CSVs
@st.cache_data(show_spinner=False)
def load_csvs():
    movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    movies['movieId'] = movies['movieId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)
    ratings['userId'] = ratings['userId'].astype(int)
    return movies, ratings

# Load Models 
@st.cache_resource(show_spinner=False)
def load_models():
    svd_model = joblib.load(Path(MODELS_DIR)/"svd_model.pkl")
    tfidf = joblib.load(Path(MODELS_DIR)/"tfidf_vectorizer.pkl")
    tfidf_matrix = sparse.load_npz(Path(MODELS_DIR)/"tfidf_matrix.npz")
    movie_idx_map = joblib.load(Path(MODELS_DIR)/"movie_idx_map.pkl")
    return svd_model, tfidf, tfidf_matrix, movie_idx_map

# Placeholder 
def load_placeholder(size=(160,240)):
    if not PLACEHOLDER_PATH.exists():
        img = Image.new("RGB", size, (200,200,200))
    else:
        img = Image.open(PLACEHOLDER_PATH).convert("RGB")
    return img.resize(size)

# Poster Fetch
def fetch_poster(title, year=None, movie_id=None, size=(160,240)):
    """Fetch poster from OMDb first, TMDb fallback, then placeholder. Cache locally."""
    if movie_id is None:
        movie_id = abs(hash(title)) % 10**8
    poster_path = POSTER_DIR / f"{movie_id}.jpg"

    # Load cached poster
    if poster_path.exists():
        try:
            img = Image.open(poster_path).convert("RGB")
            return img.resize(size)
        except: pass

    # OMDb fetch
    if OMDB_API_KEY:
        try:
            title_clean = title.rsplit('(',1)[0].strip()
            query = f"http://www.omdbapi.com/?t={quote(title_clean)}&apikey={OMDB_API_KEY}"
            if year and str(year).isdigit():
                query += f"&y={year}"
            resp = requests.get(query, timeout=5).json()
            poster_url = resp.get("Poster")
            if poster_url and poster_url!="N/A":
                rimg = requests.get(poster_url, timeout=8)
                img = Image.open(BytesIO(rimg.content)).convert("RGB")
                img = img.resize(size)
                img.save(poster_path)
                return img
        except: pass

    # TMDb fallback
    if TMDB_API_KEY:
        try:
            title_clean = title.rsplit('(',1)[0].strip()
            title_encoded = quote(title_clean)
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title_encoded}"
            if year and str(year).isdigit():
                search_url += f"&year={year}"
            resp = requests.get(search_url, timeout=5).json()
            results = resp.get("results")
            if results:
                poster_tmdb = results[0].get("poster_path")
                if poster_tmdb:
                    full_url = f"https://image.tmdb.org/t/p/w500{poster_tmdb}"
                    rimg = requests.get(full_url, timeout=8)
                    img = Image.open(BytesIO(rimg.content)).convert("RGB")
                    img = img.resize(size)
                    img.save(poster_path)
                    return img
        except: pass

    # Fallback
    return load_placeholder(size)

# Recommendation Logic
def build_cosine(tfidf_matrix):
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_content_ids(movie_id, movies, movie_idx_map, cosine_sim, top_n=10):
    if movie_id not in movie_idx_map: return []
    idx = movie_idx_map[movie_id]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: -x[1])[1:top_n+1]
    return [int(movies.iloc[i]['movieId']) for i,_ in sim_scores]

def hybrid_recommend_ids(user_id, movies, ratings, svd_model, movie_idx_map, cosine_sim, top_n=10, alpha=0.7, top_k_cf=100):
    if user_id not in ratings['userId'].unique(): return []
    rated = ratings[ratings['userId']==user_id]['movieId'].tolist()
    all_ids = movies['movieId'].tolist()
    unrated = [m for m in all_ids if m not in rated]
    preds = []
    for mid in unrated:
        try: preds.append((mid, svd_model.predict(user_id, mid).est))
        except: preds.append((mid, 0))
    preds.sort(key=lambda x:-x[1])
    top_cf = preds[:top_k_cf]

    cf_ids = [m for m,_ in top_cf]
    cf_scores = np.array([s for _,s in top_cf])
    if not rated:
        sim_scores = np.zeros_like(cf_scores)
    else:
        top_idx = [movie_idx_map.get(mid,-1) for mid in cf_ids]
        rated_idx = [movie_idx_map.get(mid,-1) for mid in rated]
        valid_top = [i for i in top_idx if i>=0]
        valid_rated = [i for i in rated_idx if i>=0]
        if valid_top and valid_rated:
            sims = cosine_sim[np.array(valid_top)][:, np.array(valid_rated)].mean(axis=1)
            sim_scores = np.zeros_like(cf_scores)
            sim_scores[:len(sims)] = sims
        else:
            sim_scores = np.zeros_like(cf_scores)
    hybrid = alpha*cf_scores + (1-alpha)*sim_scores*5.0
    top = sorted(zip(cf_ids, hybrid), key=lambda x:-x[1])[:top_n]
    return [m for m,_ in top]

# Streamlit UI 
st.set_page_config(page_title="Week 4 - Hybrid Recommender", layout="wide")
st.title("Movie Recommendation Dashboard")

# Load data and models
movies, ratings = load_csvs()
svd_model, tfidf, tfidf_matrix, movie_idx_map = load_models()
if tfidf_matrix is None or movie_idx_map is None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    genre_cols = ["unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
                  "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
                  "Romance","Sci-Fi","Thriller","War","Western"]
    for g in genre_cols:
        if g not in movies.columns: movies[g]=0
    movies['genre_str'] = movies[genre_cols].apply(lambda x:' '.join([g for g,v in zip(genre_cols,x) if v==1]), axis=1)
    movies['content'] = movies['title'].fillna('') + " " + movies['genre_str']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    sparse.save_npz(Path(MODELS_DIR)/"tfidf_matrix.npz", tfidf_matrix)
    joblib.dump(tfidf, Path(MODELS_DIR)/"tfidf_vectorizer.pkl")
    movie_idx_map = pd.Series(movies.index, index=movies['movieId']).to_dict()
    joblib.dump(movie_idx_map, Path(MODELS_DIR)/"movie_idx_map.pkl")

if "cosine" not in st.session_state:
    st.session_state["cosine"] = build_cosine(tfidf_matrix)
cosine = st.session_state["cosine"]

# Sidebar
st.sidebar.header("Settings")
rec_type = st.sidebar.radio("Type", ["Content-Based", "Hybrid"])
top_n = st.sidebar.slider("Number of movies", 5, 30, 10, 5)
tiles_per_row = st.sidebar.selectbox("Tiles per row", [3,4,5,6], index=2)


alpha = HYBRID_ALPHA

if rec_type=="Content-Based":
    seed_title = st.sidebar.selectbox("Select movie", movies['title'].unique())
    seed_id = int(movies.loc[movies['title']==seed_title,'movieId'].iloc[0])
else:
    seed_user = st.sidebar.selectbox("Select user ID", sorted(ratings['userId'].unique()))
    seed_user = int(seed_user)

# Generate recommendations
if rec_type=="Content-Based":
    rec_ids = recommend_content_ids(seed_id, movies, movie_idx_map, cosine, top_n)
else:
    rec_ids = hybrid_recommend_ids(seed_user, movies, ratings, svd_model, movie_idx_map, cosine, top_n, alpha)

# Display
TILE = (160,240)
placeholder = load_placeholder(TILE)
st.subheader("Recommended Movies")
if not rec_ids:
    st.warning("No recommendations found.")
else:
    rows = [rec_ids[i:i+tiles_per_row] for i in range(0,len(rec_ids),tiles_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for c, mid in zip(cols,row):
            try:
                m = movies[movies['movieId']==mid].iloc[0]
                title = m['title']; year = m.get('year','')
                img = fetch_poster(title, year, mid, TILE)
            except:
                title = "Unknown"; img = placeholder
            c.image(img, width=TILE[0])
            c.markdown(f"**{title} ({year})**")
