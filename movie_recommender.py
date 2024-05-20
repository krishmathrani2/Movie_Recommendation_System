import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load the data
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Merge the datasets
data = pd.merge(ratings, movies, on='movieId')

# Create a pivot table
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0
user_movie_matrix.fillna(0, inplace=True)

# Normalize the data
scaler = StandardScaler()
user_movie_matrix_scaled = scaler.fit_transform(user_movie_matrix)

# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(user_movie_matrix_scaled)

# Convert it to a DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_recommendations(user_id, genre, start_year, end_year, num_recommendations=5):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id]

    # Get the user's similarity scores
    sim_scores = cosine_sim_df[user_id]

    # Multiply the user's ratings by the similarity scores
    sim_ratings = user_movie_matrix.mul(sim_scores, axis=0)

    # Sum the scores for each movie
    movie_scores = sim_ratings.sum(axis=0)

    # Filter out the movies the user has already rated
    already_rated = user_ratings[user_ratings > 0].index
    movie_scores = movie_scores.drop(already_rated)

    # Filter movies by genre and year range
    filtered_movies = movies[(movies['year'] >= start_year) & (movies['year'] <= end_year)]
    if genre != 'All':
        filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre)]

    if filtered_movies.empty:
        return pd.Index([])  # Return an empty index if no movies match

    # Sort the scores
    filtered_movie_scores = movie_scores[filtered_movies['title']]
    filtered_movie_scores = filtered_movie_scores.sort_values(ascending=False)

    # Return the top 'num_recommendations' movies
    return filtered_movie_scores.head(num_recommendations).index

# Example usage
user_id = 1
recommendations = get_recommendations(user_id)
print("Recommended movies for User ID 1:")
print(recommendations)
