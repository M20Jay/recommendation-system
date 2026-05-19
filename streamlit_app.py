"""
Week 7 — Recommendation System
Streamlit Dashboard — Netflix Standard

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya 🇰🇪
GitHub: github.com/M20Jay
Date: May 2026
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Netflix-Standard CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0a0a;
        color: #e8e8e8;
    }

    .main { background-color: #0a0a0a; }

    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #1a0000 0%, #0a0a0a 50%, #000a1a 100%);
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        border: 1px solid #1a1a1a;
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(229,9,20,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem;
        color: #ffffff;
        letter-spacing: 3px;
        margin: 0;
        line-height: 1;
    }

    .hero-red {
        color: #e50914;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #999999;
        margin: 12px 0 4px 0;
        font-weight: 300;
    }

    .hero-author {
        font-size: 0.9rem;
        color: #e50914;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Metric Cards */
    .metric-row {
        display: flex;
        gap: 16px;
        margin: 24px 0;
    }

    .metric-card {
        background: #141414;
        border: 1px solid #222222;
        border-radius: 12px;
        padding: 20px 24px;
        flex: 1;
        transition: border-color 0.2s;
    }

    .metric-card:hover { border-color: #e50914; }

    .metric-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.2rem;
        color: #ffffff;
        letter-spacing: 2px;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
    }

    /* Movie Cards */
    .movie-row {
        display: flex;
        align-items: center;
        background: #141414;
        border: 1px solid #222222;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.2s;
    }

    .movie-row:hover {
        border-color: #e50914;
        background: #1a1a1a;
        transform: translateX(4px);
    }

    .movie-rank {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2rem;
        color: #333333;
        width: 48px;
        min-width: 48px;
    }

    .movie-title {
        flex: 1;
        font-size: 1rem;
        font-weight: 500;
        color: #ffffff;
        padding: 0 16px;
    }

    .movie-year {
        font-size: 0.8rem;
        color: #666666;
        padding: 0 8px;
    }

    .rating-pill {
        background: #e50914;
        color: white;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.8rem;
        color: #ffffff;
        letter-spacing: 2px;
        margin: 32px 0 16px 0;
        border-left: 4px solid #e50914;
        padding-left: 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #141414;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #888888;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 8px;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #e50914 !important;
        color: white !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #1a1a1a;
    }

    /* Button */
    .stButton > button {
        background: #e50914;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        width: 100%;
        font-size: 15px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: background 0.2s;
    }

    .stButton > button:hover { background: #b00710; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 32px 0 16px 0;
        border-top: 1px solid #1a1a1a;
        margin-top: 48px;
        color: #444444;
        font-size: 0.85rem;
    }

    .footer a { color: #e50914; text-decoration: none; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"


def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_recommendations(user_id, n=10):
    try:
        r = requests.post(
            f"{API_URL}/recommend",
            json={"user_id": user_id, "n": n},
            timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <span style='font-family: Bebas Neue; font-size: 1.8rem;
                     color: #e50914; letter-spacing: 3px;'>CINEAI</span>
        <div style='font-size: 0.7rem; color: #666; 
                    letter-spacing: 2px; text-transform: uppercase;
                    margin-top: 4px;'>Movie Recommender</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    health = check_health()
    if health:
        st.success("✅ API Connected")
        st.caption(f"Model: {health.get('model', 'Item-CF')}")
        st.caption(f"v{health.get('version', '1.0.0')}")
    else:
        st.error("❌ API Offline")
        st.code("uvicorn api.main:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### 🎯 Find Your Movies")

    user_id = st.number_input(
        "User ID",
        min_value=1, max_value=943,
        value=196, step=1,
        help="943 users available · Try 196, 405, 655")

    n_recs = st.slider(
        "How many recommendations?",
        min_value=5, max_value=20, value=10)

    recommend_btn = st.button("🎬 Get My Recommendations")

    st.markdown("---")
    st.markdown("### 📈 Quick Stats")

    stats = {
        "Users": "943",
        "Movies": "1,682",
        "Ratings": "100K",
        "RMSE": "0.9540",
        "P@10": "69.7%"
    }
    for k, v in stats.items():
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.caption(k)
        with col_b:
            st.caption(f"**{v}**")

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #666; line-height: 1.8;'>
        <div style='color: #e50914; font-weight: 700; 
                    font-size: 0.85rem; margin-bottom: 8px;'>
            BUILT BY</div>
        <div style='color: #ffffff; font-weight: 600;'>
            Martin James Ng'ang'a</div>
        <div>MLOps Engineer</div>
        <div>Nairobi, Kenya 🇰🇪</div>
        <div style='margin-top: 8px;'>
            Week 7 · 15-Week MLOps Programme</div>
        <div style='margin-top: 4px;'>
            <a href='https://github.com/M20Jay' 
               style='color: #e50914;'>github.com/M20Jay</a></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">CINE<span class="hero-red">AI</span></div>
    <div class="hero-subtitle">
        Personalised Movie Recommendations · 
        Item-Based Collaborative Filtering · 
        MovieLens 100K
    </div>
    <div class="hero-author">
        🇰🇪 Built by Martin James Ng'ang'a · 
        MLOps Engineer · github.com/M20Jay · 
        Week 7 of 15-Week MLOps Programme
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">943</div>
        <div class="metric-label">Total Users</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">1,682</div>
        <div class="metric-label">Total Movies</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">100K</div>
        <div class="metric-label">Total Ratings</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">0.9540</div>
        <div class="metric-label">Model RMSE</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">69.7%</div>
        <div class="metric-label">Precision @ 10</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯 Recommendations",
    "📊 Data Insights",
    "🏆 Model Performance"
])

# ─────────────────────────────────────────────
# TAB 1 — Recommendations
# ─────────────────────────────────────────────
with tab1:
    if recommend_btn:
        with st.spinner("🎬 Finding your perfect movies..."):
            data = get_recommendations(user_id, n_recs)

        if data and data.get('recommendations'):
            recs = data['recommendations']

            st.markdown(
                f'<div class="section-header">'
                f'TOP {n_recs} FOR USER {user_id}'
                f'</div>', unsafe_allow_html=True)

            st.caption(
                f"Model: {data.get('model', 'Item-CF')} · "
                f"Generated: {data.get('timestamp', '')[:19]}")

            # Movie cards
            for i, rec in enumerate(recs, 1):
                title = rec['title']
                year = ""
                if '(' in title and ')' in title:
                    year = title[title.rfind('(')+1:title.rfind(')')]
                    title = title[:title.rfind('(')].strip()

                rating = rec['predicted_rating']
                stars = "★" * int(round(rating)) + \
                        "☆" * (5 - int(round(rating)))

                st.markdown(f"""
                <div class="movie-row">
                    <div class="movie-rank">{i:02d}</div>
                    <div class="movie-title">{title}</div>
                    <div class="movie-year">{year}</div>
                    <div style="color: #f5c518; 
                                font-size: 0.9rem; 
                                padding: 0 12px;">{stars}</div>
                    <div class="rating-pill">⭐ {rating}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Chart
            df_recs = pd.DataFrame(recs)
            df_recs['short_title'] = df_recs['title'].apply(
                lambda x: x[:25] + '...' if len(x) > 25 else x)

            fig = px.bar(
                df_recs,
                x='predicted_rating',
                y='short_title',
                orientation='h',
                color='predicted_rating',
                color_continuous_scale=['#330000', '#e50914'],
                range_x=[0, 5],
                title=f"Predicted Ratings — User {user_id}"
            )
            fig.update_layout(
                plot_bgcolor='#141414',
                paper_bgcolor='#0a0a0a',
                font_color='#cccccc',
                title_font_color='#ffffff',
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

            # Download
            st.download_button(
                "📥 Download Recommendations",
                data=pd.DataFrame(recs).to_csv(index=False),
                file_name=f"recommendations_user_{user_id}.csv",
                mime="text/csv")

        else:
            st.error(
                f"Could not get recommendations for User {user_id}.")

    else:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;
                    color: #444444;'>
            <div style='font-size: 4rem;'>🎬</div>
            <div style='font-family: Bebas Neue; font-size: 1.8rem;
                        letter-spacing: 2px; margin: 16px 0 8px 0;
                        color: #666666;'>
                SELECT A USER AND CLICK GET RECOMMENDATIONS
            </div>
            <div style='font-size: 0.9rem; color: #444444;'>
                Try User ID 196, 405, or 655 to start
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2 — Data Insights
# ─────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-header">DATA INSIGHTS</div>',
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Rating distribution
        rating_data = pd.DataFrame({
            'Rating': [1, 2, 3, 4, 5],
            'Count': [6110, 11370, 27145, 34174, 21201],
            'Percentage': [6.1, 11.4, 27.1, 34.2, 21.2]
        })

        fig1 = px.bar(
            rating_data,
            x='Rating', y='Count',
            color='Rating',
            color_continuous_scale=['#330000', '#e50914', '#ff6b6b'],
            title='Rating Distribution — MovieLens 100K',
            text='Percentage'
        )
        fig1.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside')
        fig1.update_layout(
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            font_color='#cccccc',
            title_font_color='#ffffff',
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Genre popularity
        genre_data = pd.DataFrame({
            'Genre': ['Drama', 'Comedy', 'Action',
                      'Thriller', 'Romance', 'Adventure',
                      'Crime', 'Sci-Fi', 'Horror'],
            'Movies': [725, 505, 251, 251, 247, 135, 109, 101, 92]
        })

        fig2 = px.bar(
            genre_data,
            x='Movies', y='Genre',
            orientation='h',
            color='Movies',
            color_continuous_scale=['#1a0000', '#e50914'],
            title='Movies per Genre'
        )
        fig2.update_layout(
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            font_color='#cccccc',
            title_font_color='#ffffff',
            coloraxis_showscale=False,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=40, b=20),
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Sparsity donut
        fig3 = go.Figure(data=[go.Pie(
            labels=['Rated', 'Unrated'],
            values=[6.3, 93.7],
            hole=0.7,
            marker_colors=['#e50914', '#1a1a1a'],
            textinfo='label+percent'
        )])
        fig3.update_layout(
            title='Matrix Sparsity — 93.7% Empty',
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            font_color='#cccccc',
            title_font_color='#ffffff',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            showlegend=False,
            annotations=[dict(
                text='93.7%<br>Sparse',
                x=0.5, y=0.5,
                font_size=16,
                font_color='#ffffff',
                showarrow=False)]
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Gender split
        fig4 = go.Figure(data=[go.Pie(
            labels=['Male', 'Female'],
            values=[71, 29],
            hole=0.7,
            marker_colors=['#4f8ef7', '#e50914'],
            textinfo='label+percent'
        )])
        fig4.update_layout(
            title='User Gender Distribution',
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            font_color='#cccccc',
            title_font_color='#ffffff',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            showlegend=False,
            annotations=[dict(
                text='71% Male',
                x=0.5, y=0.5,
                font_size=14,
                font_color='#ffffff',
                showarrow=False)]
        )
        st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 3 — Model Performance
# ─────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="section-header">MODEL PERFORMANCE</div>',
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # RMSE comparison
        model_data = pd.DataFrame({
            'Model': ['User-CF', 'SVD', 'Item-CF'],
            'RMSE': [0.9703, 0.9561, 0.9540],
            'Production': [False, False, True]
        })

        fig5 = px.bar(
            model_data,
            x='Model', y='RMSE',
            color='Production',
            color_discrete_map={True: '#e50914', False: '#333333'},
            title='RMSE Comparison — Lower is Better',
            text='RMSE'
        )
        fig5.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside')
        fig5.update_layout(
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            font_color='#cccccc',
            title_font_color='#ffffff',
            showlegend=False,
            yaxis_range=[0.90, 0.98],
            margin=dict(l=20, r=20, t=40, b=20),
            height=350
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        # MAE comparison
        fig6 = px.bar(
            model_data,
            x='Model',
            y=[0.7654, 0.7524, 0.7488],
            color='Production',
            color_discrete_map={True: '#e50914', False: '#333333'},
            title='MAE Comparison — Lower is Better',
            text=[0.7654, 0.7524, 0.7488]
        )
        fig6.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside')
        fig6.update_layout(
            plot_bgcolor='#141414',
            paper_bgcolor='#141414',
            font_color='#cccccc',
            title_font_color='#ffffff',
            showlegend=False,
            yaxis_range=[0.70, 0.78],
            margin=dict(l=20, r=20, t=40, b=20),
            height=350
        )
        st.plotly_chart(fig6, use_container_width=True)

    # Model cards
    st.markdown(
        '<div class="section-header">MODEL DETAILS</div>',
        unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div style='background: #141414; border: 1px solid #222;
                    border-radius: 12px; padding: 24px;'>
            <div style='font-size: 0.75rem; color: #666;
                        letter-spacing: 2px; text-transform: uppercase;
                        margin-bottom: 8px;'>SVD</div>
            <div style='font-size: 1.1rem; color: #fff;
                        font-weight: 600; margin-bottom: 16px;'>
                Matrix Factorisation</div>
            <div style='color: #999; font-size: 0.85rem;
                        line-height: 2;'>
                RMSE: 0.9561<br>
                MAE: 0.7524<br>
                n_factors: 100<br>
                n_epochs: 20
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background: #1a0000; border: 2px solid #e50914;
                    border-radius: 12px; padding: 24px;'>
            <div style='font-size: 0.75rem; color: #e50914;
                        letter-spacing: 2px; text-transform: uppercase;
                        margin-bottom: 8px;'>✅ PRODUCTION</div>
            <div style='font-size: 1.1rem; color: #fff;
                        font-weight: 600; margin-bottom: 16px;'>
                Item-Based CF</div>
            <div style='color: #999; font-size: 0.85rem;
                        line-height: 2;'>
                RMSE: 0.9540 ← Best<br>
                MAE: 0.7488 ← Best<br>
                P@10: 69.7%<br>
                k: 40 · cosine similarity
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div style='background: #141414; border: 1px solid #222;
                    border-radius: 12px; padding: 24px;'>
            <div style='font-size: 0.75rem; color: #666;
                        letter-spacing: 2px; text-transform: uppercase;
                        margin-bottom: 8px;'>USER-CF</div>
            <div style='font-size: 1.1rem; color: #fff;
                        font-weight: 600; margin-bottom: 16px;'>
                User-Based CF</div>
            <div style='color: #999; font-size: 0.85rem;
                        line-height: 2;'>
                RMSE: 0.9703<br>
                MAE: 0.7654<br>
                k: 40<br>
                cosine similarity
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong style='color: #ffffff;'>CineAI</strong> · 
    Built by 
    <strong style='color: #e50914;'>Martin James Ng'ang'a</strong> · 
    MLOps Engineer · Nairobi, Kenya 🇰🇪 · 
    Week 7 of 15-Week MLOps Programme · 
    <a href='https://github.com/M20Jay'>github.com/M20Jay</a>
</div>
""", unsafe_allow_html=True)