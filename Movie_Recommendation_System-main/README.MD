# 🎬Movie Recommendation System

## 📖 Overview
This project aims to build a **Movie Recommendation System** using the **MovieLens 100k (ml-100k) dataset**.  
The dataset contains **100,000 ratings** from **943 users** on **1,682 movies**.  
The goal is to analyze user–movie interactions and recommend movies based on user preferences.

## 🧠 Project Objective
- Develop a recommendation engine capable of predicting how a user would rate a movie they haven’t yet watched.
- Combine **Content-Based Filtering** and **Collaborative Filtering** techniques to provide more accurate and diverse recommendations.
- Build an interactive **Streamlit web app** for real-time movie recommendations.

---

## 📊 Dataset Details
**Dataset:** [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/)  
**Files Used:**
- `u.data` → User–Movie ratings  
- `u.item` → Movie metadata (titles, genres, release dates)  
- `u.user` → User demographics (age, gender, occupation, etc.)


## 🧩 Techniques Used
### 🔹 Content-Based Filtering
- Utilizes **TF-IDF Vectorization** on movie descriptions or genres.
- Calculates **cosine similarity** between movies.
- Recommends movies similar to those the user has liked.

### 🔹 Collaborative Filtering
- Uses **Singular Value Decomposition (SVD)** algorithm from **Surprise library**.
- Learns latent features representing user and movie characteristics.
- Predicts ratings for unseen movies based on past behavior.
