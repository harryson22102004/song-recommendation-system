import streamlit as st
from recommender import SongRecommender

st.set_page_config(page_title="Song Recommendation System", layout="centered")

st.title("🎵 Song Recommendation System")

recommender = SongRecommender("songs.csv")

song_list = recommender.df['title'].values

selected_song = st.selectbox("Select a song", song_list)

if st.button("Recommend"):
    recommendations = recommender.recommend(selected_song)

    st.subheader("Recommended Songs:")
    for song in recommendations:
        st.write("•", song)
