import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

data = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

scaler = StandardScaler()
user_movie_matrix_scaled = scaler.fit_transform(user_movie_matrix)


cosine_sim_matrix = cosine_similarity(user_movie_matrix_scaled)

cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_recommendations(user_id, genre, start_year, end_year, num_recommendations=5):
    user_ratings = user_movie_matrix.loc[user_id]
    sim_scores = cosine_sim_df[user_id]

    sim_ratings = user_movie_matrix.mul(sim_scores, axis=0)
    movie_scores = sim_ratings.sum(axis=0)

    already_rated = user_ratings[user_ratings > 0].index
    movie_scores = movie_scores.drop(already_rated)

    filtered_movies = movies[(movies['year'] >= start_year) & (movies['year'] <= end_year)]
    if genre != 'All':
        filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre)]

    if filtered_movies.empty:
        return pd.Index([]) 

    filtered_movie_scores = movie_scores[filtered_movies['title']]
    filtered_movie_scores = filtered_movie_scores.sort_values(ascending=False)

    return filtered_movie_scores.head(num_recommendations).index

user_id = 1
recommendations = get_recommendations(user_id)
print("Recommended movies for User ID 1:")
print(recommendations)
