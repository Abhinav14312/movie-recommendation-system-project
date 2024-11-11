import streamlit as st
import pandas as pd
import difflib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to fetch movie poster using TMDb API
def fetch_poster(movie_title, api_key):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    try:
        response = requests.get(search_url)
        data = response.json()

        if data['results']:
            poster_path = data['results'][0]['poster_path']
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return None  # No poster found
    except Exception as e:
        st.write(f"Error fetching poster for {movie_title}: {e}")
        return None

# Load the data
movies_data = pd.read_csv("C:/Users/abhia/OneDrive/Desktop/Movie recommendation system/moviedata.csv")

# Prepare the data for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'popularity', 'vote_average']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = (
    movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + 
    movies_data['tagline'] + ' ' + movies_data['popularity'].astype(str) + 
    ' ' + movies_data['vote_average'].astype(str)
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Recommendation function
def recommend_movies(movie_name, api_key):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return ["No matching movie found."]
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommended_movies = []
    posters = []
    for i, movie in enumerate(sorted_similar_movies[1:31], start=1):
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        recommended_movies.append(title_from_index)
        
        # Fetch poster for the movie
        poster_url = fetch_poster(title_from_index, api_key)
        posters.append(poster_url)
        
    return recommended_movies, posters

# Streamlit UI
st.title("Movie Recommendation System")
movie_name = st.text_input("Enter your favorite movie:")

# API Key for TMDb (replace with your own key)
api_key = "YOUR_TMDB_API_KEY"  # Replace this with your actual TMDb API key

if st.button("Recommend"):
    if movie_name:
        recommendations, posters = recommend_movies(movie_name, api_key)
        st.write("Movies suggested for you:")
        for i, (title, poster_url) in enumerate(zip(recommendations, posters), start=1):
            st.write(f"{i}. {title}")
            if poster_url:
                st.image(poster_url, width=150)
            else:
                st.write(f"Poster not available for {title}")
    else:
        st.write("Please enter a movie name.")
