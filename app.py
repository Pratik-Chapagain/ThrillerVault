import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ThrillerVault — AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STYLING
# ============================================================
st.markdown("""
<style>
    /* ---------- App ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(124, 92, 255, 0.08), transparent 28%),
            radial-gradient(circle at 90% 5%, rgba(236, 72, 153, 0.06), transparent 25%),
            #09090f;
        color: #f4f4f5;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 9, 15, 0.82);
    }

    [data-testid="stSidebar"] {
        background: #0e0e16;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] * {
        color: #e8e8ef;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------- Hero ---------- */
    .hero {
        padding: 1.2rem 0 2rem 0;
    }

    .eyebrow {
        color: #a78bfa;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: clamp(2.7rem, 5vw, 5rem);
        line-height: 0.98;
        font-weight: 900;
        letter-spacing: -0.055em;
        margin: 0;
        background: linear-gradient(100deg, #ffffff 15%, #c4b5fd 55%, #f9a8d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-copy {
        max-width: 720px;
        color: #a1a1aa;
        font-size: 1.08rem;
        line-height: 1.7;
        margin-top: 1rem;
    }

    /* ---------- Section headers ---------- */
    .section-kicker {
        color: #a78bfa;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .section-title {
        color: #fafafa;
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }

    .muted {
        color: #8f8f9b;
        font-size: 0.92rem;
    }

    /* ---------- Cards ---------- */
    .glass-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.15rem;
        box-shadow: 0 18px 50px rgba(0,0,0,0.16);
    }

    .selected-card {
        background: linear-gradient(135deg, rgba(124,92,255,0.16), rgba(236,72,153,0.07));
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 22px;
        padding: 1.35rem 1.5rem;
        margin: 1.1rem 0 1.7rem;
    }

    .movie-meta {
        color: #a1a1aa;
        font-size: 0.82rem;
        line-height: 1.65;
    }

    .movie-name {
        color: #fafafa;
        font-size: 1.08rem;
        font-weight: 800;
        line-height: 1.25;
        margin: 0.55rem 0 0.4rem;
    }

    .poster-wrap {
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 0.55rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: rgba(167,139,250,0.14);
        color: #c4b5fd;
        border: 1px solid rgba(167,139,250,0.22);
        font-size: 0.72rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }

    /* ---------- Search panel ---------- */
    .search-panel {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1.25rem 1.35rem;
        margin: 1.2rem 0;
    }

    .example-box {
        background: rgba(124,92,255,0.08);
        border: 1px solid rgba(167,139,250,0.16);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        color: #b7b7c2;
        font-size: 0.88rem;
        line-height: 1.65;
        margin: 0.8rem 0 1rem;
    }

    /* ---------- Metrics ---------- */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 0.85rem 1rem;
    }

    .metric-label {
        color: #8f8f9b;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 750;
    }

    .metric-value {
        color: #fafafa;
        font-size: 1.25rem;
        font-weight: 850;
        margin-top: 0.18rem;
    }

    /* ---------- Sidebar ---------- */
    .side-brand {
        padding: 0.5rem 0 1.2rem;
    }

    .side-brand-title {
        font-size: 1.3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
    }

    .side-brand-sub {
        color: #777784;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    .side-heading {
        color: #8f8f9b;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 1.2rem 0 0.65rem;
    }

    .insight {
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.055);
    }

    .insight:last-child {
        border-bottom: 0;
    }

    .insight-label {
        color: #777784;
        font-size: 0.72rem;
    }

    .insight-value {
        color: #ededf2;
        font-size: 0.84rem;
        font-weight: 700;
        margin-top: 0.12rem;
    }

    /* ---------- Streamlit controls ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(255,255,255,0.055);
        border-color: rgba(255,255,255,0.10);
        border-radius: 12px;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="input"] input {
        color: #f4f4f5;
    }

    label, .stSlider label, .stSelectbox label, .stTextInput label {
        color: #c9c9d2 !important;
        font-weight: 650 !important;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.055);
        color: #f4f4f5;
        font-weight: 700;
        transition: all .18s ease;
    }

    .stButton > button:hover {
        border-color: rgba(167,139,250,0.5);
        background: rgba(167,139,250,0.12);
        transform: translateY(-1px);
    }

    /* ---------- Result rows ---------- */
    .result-row {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }

    .rank {
        color: #a78bfa;
        font-weight: 900;
        font-size: 1.1rem;
    }

    /* ---------- Footer ---------- */
    .footer {
        border-top: 1px solid rgba(255,255,255,0.07);
        margin-top: 3rem;
        padding-top: 1.2rem;
        text-align: center;
        color: #686875;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA + MODELS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/thriller_movies_enriched.csv")

    for col in ["plot", "director", "genres", "title"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df["decade"] = (df["release_year"] // 10) * 10
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["box_office_usd"] = pd.to_numeric(df["box_office_usd"], errors="coerce")

    return df


@st.cache_resource
def load_models(df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(df["plot"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["plot"].tolist(),
        show_progress_bar=False
    )

    return cosine_sim, embeddings, model


df = load_data()

with st.spinner("Preparing ThrillerVault AI..."):
    cosine_sim, embeddings, model = load_models(df)


# ============================================================
# HELPERS
# ============================================================
def safe_text(value, fallback="N/A"):
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value)


def movie_card(movie, score=None):
    poster = movie.get("poster_url")

    if pd.notna(poster) and str(poster).strip():
        st.image(poster, use_container_width=True)

    st.markdown(
        f'<div class="movie-name">{safe_text(movie["title"])}</div>',
        unsafe_allow_html=True
    )

    rating = (
        f'⭐ {movie["vote_average"]:.1f}'
        if pd.notna(movie.get("vote_average"))
        else "⭐ N/A"
    )

    runtime = (
        f' · {int(movie["runtime"])} min'
        if pd.notna(movie.get("runtime"))
        else ""
    )

    st.markdown(
        f'<div class="movie-meta">📅 {safe_text(movie["release_year"])} · '
        f'{rating}{runtime}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="movie-meta">🎬 {safe_text(movie["director"])}</div>',
        unsafe_allow_html=True
    )

    genres = safe_text(movie["genres"])
    st.markdown(
        f'<div class="movie-meta">🎞️ {genres[:70]}</div>',
        unsafe_allow_html=True
    )

    if score is not None:
        st.markdown(
            f'<span class="badge">AI match · {score:.0%}</span>',
            unsafe_allow_html=True
        )


def section_header(kicker, title, description=""):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="muted">{description}</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="side-brand">
        <div class="side-brand-title">🎬 ThrillerVault</div>
        <div class="side-brand-sub">AI-powered movie discovery</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-heading">Explore</div>', unsafe_allow_html=True)

    search_type = st.radio(
        "Navigation",
        [
            "🎯 Similar Movies",
            "🧠 AI Search",
            "📊 Explore Library",
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="side-heading">Library</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Movies</div>'
            f'<div class="metric-value">{len(df):,}</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        years = f'{int(df["release_year"].min())}–{int(df["release_year"].max())}'
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Years</div>'
            f'<div class="metric-value">{years}</div></div>',
            unsafe_allow_html=True
        )

    top_director = (
        df["director"].replace("", np.nan).dropna().value_counts().index[0]
        if not df["director"].replace("", np.nan).dropna().empty
        else "N/A"
    )

    top_rated = (
        df.loc[df["vote_average"].idxmax(), "title"]
        if df["vote_average"].notna().any()
        else "N/A"
    )

    highest_grossing = (
        df.loc[df["box_office_usd"].idxmax(), "title"]
        if df["box_office_usd"].notna().any()
        else "N/A"
    )

    st.markdown('<div class="side-heading">Quick insights</div>', unsafe_allow_html=True)

    for label, value in [
        ("Most represented director", top_director),
        ("Highest rated", top_rated),
        ("Highest grossing", highest_grossing),
    ]:
        st.markdown(
            f'<div class="insight"><div class="insight-label">{label}</div>'
            f'<div class="insight-value">{value}</div></div>',
            unsafe_allow_html=True
        )


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AI MOVIE DISCOVERY</div>
    <div class="hero-title">Find your next<br>thriller.</div>
    <div class="hero-copy">
        Discover movies by similarity, describe exactly what you want,
        or explore the entire thriller library.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 1. SIMILAR MOVIES
# ============================================================
if search_type == "🎯 Similar Movies":

    section_header(
        "DISCOVER",
        "Find movies like your favorite",
        "Choose one movie and let the recommendation engine do the work."
    )

    st.markdown('<div class="search-panel">', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])

    with col1:
        selected_movie = st.selectbox(
            "Movie",
            sorted(df["title"].tolist()),
            label_visibility="visible"
        )

    with col2:
        top_n = st.slider(
            "Recommendations",
            4,
            12,
            8
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if selected_movie:
        idx = df.index[df["title"] == selected_movie][0]
        movie_info = df.loc[idx]

        rating = (
            f'{movie_info["vote_average"]:.1f}/10'
            if pd.notna(movie_info["vote_average"])
            else "N/A"
        )

        runtime = (
            f'{int(movie_info["runtime"])} min'
            if pd.notna(movie_info["runtime"])
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="selected-card">
                <div class="section-kicker">YOUR SELECTION</div>
                <div class="section-title">{selected_movie}</div>
                <div class="muted">
                    {safe_text(movie_info["director"])} ·
                    {safe_text(movie_info["release_year"])} ·
                    ⭐ {rating} · ⏱️ {runtime}
                </div>
                <p style="color:#b5b5bf; line-height:1.7; margin-bottom:0.4rem;">
                    {safe_text(movie_info["plot"])}
                </p>
                <span class="badge">{safe_text(movie_info["genres"])}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        section_header(
            "RECOMMENDATIONS",
            f"Because you liked {selected_movie}",
            "Movies ranked by plot similarity."
        )

        st.write("")

        cols_per_row = 4

        for row_start in range(0, len(sim_scores), cols_per_row):
            cols = st.columns(cols_per_row)

            for col, (movie_idx, score) in zip(
                cols,
                sim_scores[row_start:row_start + cols_per_row]
            ):
                movie = df.iloc[movie_idx]

                with col:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    movie_card(movie, score)
                    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 2. NATURAL LANGUAGE AI SEARCH
# ============================================================
elif search_type == "🧠 AI Search":

    section_header(
        "NATURAL LANGUAGE",
        "Tell the AI what you want",
        "Describe a mood, story, character, or type of thriller in your own words."
    )

    st.markdown("""
    <div class="example-box">
        <b>Try:</b>
        psychological thriller about a detective ·
        dark crime mystery with a huge twist ·
        supernatural thriller set in an isolated town ·
        intelligent serial-killer investigation
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])

    with col1:
        query = st.text_input(
            "Search",
            placeholder="Describe the movie you're in the mood for...",
            key="nl_query"
        )

    with col2:
        top_n = st.slider("Results", 3, 10, 6, key="nl_slider")

    if query.strip():

        with st.spinner("Understanding your request..."):
            query_embedding = model.encode([query])
            similarities = cosine_similarity(
                query_embedding,
                embeddings
            )[0]

            similar_indices = similarities.argsort()[-top_n:][::-1]

        st.write("")

        section_header(
            "AI RESULTS",
            f"Top {top_n} matches",
            f"Based on: “{query}”"
        )

        st.write("")

        for rank, idx in enumerate(similar_indices, 1):
            movie = df.iloc[idx]
            similarity = similarities[idx]

            st.markdown('<div class="result-row">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2.8, 2])

            with col1:
                poster = movie.get("poster_url")
                if pd.notna(poster) and str(poster).strip():
                    st.image(poster, use_container_width=True)

            with col2:
                st.markdown(
                    f'<div class="rank">#{rank}</div>'
                    f'<div class="movie-name">{movie["title"]}</div>'
                    f'<div class="movie-meta">'
                    f'{safe_text(movie["release_year"])} · '
                    f'🎬 {safe_text(movie["director"])}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if pd.notna(movie.get("vote_average")):
                    st.markdown(
                        f'<div class="movie-meta">⭐ '
                        f'{movie["vote_average"]:.1f}/10</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f'<div class="movie-meta">🎞️ '
                    f'{safe_text(movie["genres"])}</div>',
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f'<span class="badge">AI match · {similarity:.0%}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p class="muted" style="line-height:1.6;">'
                    f'{safe_text(movie["plot"])[:240]}...</p>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 3. EXPLORE LIBRARY
# ============================================================
else:

    section_header(
        "LIBRARY",
        "Explore every thriller",
        "Filter the collection by era, director, rating, and sort order."
    )

    st.markdown('<div class="search-panel">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        decades = sorted(df["decade"].dropna().unique())
        selected_decade = st.selectbox(
            "Decade",
            ["All"] + [f"{int(d)}s" for d in decades]
        )

    with col2:
        directors = sorted(
            df["director"].replace("", np.nan).dropna().unique()
        )
        selected_director = st.selectbox(
            "Director",
            ["All"] + directors
        )

    with col3:
        min_rating = st.slider(
            "Minimum rating",
            0.0,
            10.0,
            5.0,
            0.5
        )

    with col4:
        sort_by = st.selectbox(
            "Sort by",
            ["Title", "Year", "Rating", "Box Office"]
        )

    st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df.copy()

    if selected_decade != "All":
        decade_num = int(selected_decade.replace("s", ""))
        filtered_df = filtered_df[
            filtered_df["decade"] == decade_num
        ]

    if selected_director != "All":
        filtered_df = filtered_df[
            filtered_df["director"] == selected_director
        ]

    filtered_df = filtered_df[
        filtered_df["vote_average"].fillna(0) >= min_rating
    ]

    sort_map = {
        "Title": "title",
        "Year": "release_year",
        "Rating": "vote_average",
        "Box Office": "box_office_usd"
    }

    filtered_df = filtered_df.sort_values(
        sort_map[sort_by],
        ascending=(sort_by == "Title"),
        na_position="last"
    )

    st.markdown(
        f'<div class="muted"><b>{len(filtered_df):,}</b> movies found</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if filtered_df.empty:
        st.info("No movies match these filters. Try lowering the rating or changing the filters.")
    else:
        cols_per_row = 4

        for row_start in range(0, len(filtered_df), cols_per_row):
            cols = st.columns(cols_per_row)

            for col, (_, movie) in zip(
                cols,
                filtered_df.iloc[row_start:row_start + cols_per_row].iterrows()
            ):
                with col:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    movie_card(movie)
                    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    ThrillerVault · AI-powered thriller discovery
</div>
""", unsafe_allow_html=True)