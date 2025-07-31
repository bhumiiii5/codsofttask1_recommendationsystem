import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# 🔧 Load only top 1000 movies to save memory
movies = pd.read_csv("movies.csv").head(1000)

# 🧼 Clean genre data: replace '|' with space and fill missing values
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# 🔍 Vectorize genres with limited features to reduce memory
vectorizer = TfidfVectorizer(max_features=1000)
genre_vectors = vectorizer.fit_transform(movies['genres'])

# ✅ Compute similarity matrix
similarity = cosine_similarity(genre_vectors)

# 🔁 Recommendation function
def recommend(movie_title, top_n=5):
    try:
        # Get the index of the movie by partial title match
        idx = movies[movies['title'].str.contains(movie_title, case=False, na=False)].index[0]
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]
        return movies.iloc[top_indices]['title'].tolist()
    except IndexError:
        return []

# 🎨 GUI logic
def get_recommendations():
    movie_name = entry.get()
    if not movie_name.strip():
        messagebox.showwarning("Input Error", "Please enter a movie name.")
        return
    results = recommend(movie_name)
    if results:
        output_text.set("Recommended:\n" + "\n".join(results))
    else:
        output_text.set("No recommendations found.\nTry another movie.")

# 🖼️ Build the GUI window
root = tk.Tk()
root.title("Movie Recommendation System")

tk.Label(root, text="Enter Movie Name:", font=('Arial', 14)).pack(pady=10)
entry = tk.Entry(root, width=40, font=('Arial', 12))
entry.pack()

tk.Button(root, text="Get Recommendations", command=get_recommendations).pack(pady=10)

output_text = tk.StringVar()
tk.Label(root, textvariable=output_text, font=('Arial', 12), wraplength=400, justify='left').pack(pady=10)

root.mainloop()



