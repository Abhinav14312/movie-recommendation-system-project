import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
def recommend_movies(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return ["No matching movie found."]
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:31], start=1):
        index = movie[0]
        title_from_index = movies_data.iloc[index]['title']
        recommended_movies.append(title_from_index)
    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")
movie_name = st.text_input("Enter your favorite movie:")

if st.button("Recommend"):
    if movie_name:
        recommendations = recommend_movies(movie_name)
        st.write("Movies suggested for you:")
        for i, title in enumerate(recommendations, start=1):
            st.write(f"{i}. {title}")
    else:
        st.write("Please enter a movie name.")
