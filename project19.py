# movie_recommender_gui.py
# Interactive Movie Recommendation System with Tkinter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk

# -----------------------------
# Load Dataset
# -----------------------------
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
movies = movies.dropna(subset=['movieId'])
ratings = ratings.dropna(subset=['movieId'])

movies['genres'] = movies['genres'].fillna('')
movies['genres_str'] = movies['genres'].str.replace('|', ' ')

# -----------------------------
# Create Similarity Matrix
# -----------------------------
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['genres_str'])
cosine_sim = cosine_similarity(count_matrix)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("Movie Recommender System")
root.geometry("500x400")
root.config(bg="#f0f0f0")

# Title Label
title_label = tk.Label(root, text="Movie Recommendation System", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

# Entry for movie name
entry_label = tk.Label(root, text="Enter a Movie Name:", bg="#f0f0f0")
entry_label.pack(pady=5)
movie_entry = tk.Entry(root, width=50)
movie_entry.pack(pady=5)

# Listbox to show recommendations
listbox = tk.Listbox(root, width=50, height=10)
listbox.pack(pady=10)

# Function to show recommendations
def show_recommendations():
    movie_name = movie_entry.get()
    listbox.delete(0, tk.END)
    recommendations = recommend(movie_name)
    if not recommendations:
        messagebox.showerror("Error", "Movie not found in dataset!")
    else:
        for i, movie in enumerate(recommendations, start=1):
            listbox.insert(tk.END, f"{i}. {movie}")

# Recommend Button
recommend_button = tk.Button(root, text="Get Recommendations", command=show_recommendations, bg="#4CAF50", fg="white")
recommend_button.pack(pady=10)

# Start GUI
root.mainloop()
