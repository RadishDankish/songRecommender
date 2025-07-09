import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import streamlit as st

df =  pd.read_csv('SpotifyFeatures.csv')

df_new   = df[['genre','track_name','popularity',
       'acousticness', 'danceability', 'duration_ms', 'energy',
       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence']]


df_modified = pd.get_dummies(df_new, columns=['genre', 'key','mode','time_signature'],dtype='int')
df_modified.drop(['duration_ms'], axis=1, inplace=True)

scaler = MinMaxScaler()
columns_to_scale = ['popularity', 'acousticness', 'danceability', 'energy',
       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
       'valence']

df_modified[columns_to_scale] = scaler.fit_transform(df_modified[columns_to_scale])
df_modified.drop(['popularity'], axis=1, inplace=True)
columns_used = df_modified.columns.difference(['track_name'])
X = df_modified[columns_used]
y = df_modified['track_name']



model = NearestNeighbors(n_neighbors=10,metric='cosine', algorithm='brute')
model.fit(X)


def recommend_songs(song_name,number_of_songs=10):
    if song_name not in y.values:
        print("Song not found in the dataset.")
        return
    
    song_index = y[y == song_name].index[0]
    
    distances,indices = model.kneighbors(X.iloc[song_index].values.reshape(1,-1),n_neighbors=number_of_songs)
    
    recommended_songs = []
    for song in indices[0]:
        recommended_songs.append(y.iloc[song])
    
    return recommended_songs
        


st.title("🎵 Song Recommender")

song = st.selectbox("Choose a song:", y.values)

if st.button("Recommend"):
    recs = recommend_songs(song)
    st.write("Top 5 similar songs:")
    for r in recs:
        st.write(f"- {r}")