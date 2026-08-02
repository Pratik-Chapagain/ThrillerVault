# 🎬 ThrillerVault

### AI-Powered Movie Discovery & Recommendation System

ThrillerVault is an AI-powered movie discovery system built around a curated dataset of **80 thriller films from 1950–2026**.

It combines **data analysis, NLP, machine learning, and semantic search** to recommend movies and find films using natural-language descriptions.

## ✨ Features

* 🎬 80 curated thriller movies
* 📊 Movie data analysis and visualizations
* 🤖 TF-IDF + Cosine Similarity recommendations
* 🧠 Sentence-Transformer semantic embeddings
* 🔎 Natural-language movie search
* 🖼️ Movie posters, ratings, and metadata
* 🌐 Interactive Streamlit web app

## 🧠 How It Works

```text
Movie Dataset
     ↓
Data Cleaning & Analysis
     ↓
TF-IDF / Embeddings
     ↓
Similarity Search
     ↓
Movie Recommendations
     ↓
Streamlit App
```

## 🛠️ Tech Stack

**Python · Pandas · NumPy · Scikit-learn · Sentence-Transformers · Matplotlib · Streamlit · Requests**

## 📁 Project Structure

```text
ThrillerVault/
├── data/
│   ├── thriller_movies.csv
│   └── thriller_movies_enriched.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── clean_data.py
│   └── recommender.py
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ThrillerVault

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

## 📊 Data

The dataset was manually curated and enriched using movie metadata sources including **TMDB** and **IMDb**.

> This is an educational prototype and the dataset is not intended to be a comprehensive movie database.

## 🔮 Future Plans

* Expand dataset to 500+ movies
* Improve recommendation accuracy
* Add personalized recommendations
* Add collaborative filtering
* Deploy the application

## 👨‍💻 Author

**Pratik Chapagain**

BSc CSIT Student | AGENTIC AI & AUTOMATION ENTHUSIAST

---

⭐ **Built to learn AI by building something I genuinely enjoy.**
