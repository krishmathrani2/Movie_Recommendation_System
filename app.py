from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the data
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
genres = movies['genres'].str.get_dummies(sep='|')

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

    # Sort the scores
    filtered_movie_scores = movie_scores[filtered_movies['title']]
    filtered_movie_scores = filtered_movie_scores.sort_values(ascending=False)

    # Return the top 'num_recommendations' movies
    return filtered_movie_scores.head(num_recommendations).index

@app.route('/')
def index():
    genres_list = genres.columns.tolist()
    genres_list.insert(0, 'All')
    return render_template('index.html', genres=genres_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    print("Form Data:", request.form)  # Debugging: print the form data
    user_id = int(request.form['user_id'])
    genre = request.form['genre']
    start_year = int(request.form['start_year'])
    end_year = int(request.form['end_year'])
    num_recommendations = int(request.form['num_recommendations'])
    recommendations = get_recommendations(user_id, genre, start_year, end_year, num_recommendations)
    recommendations_list = recommendations.tolist() if not recommendations.empty else []
    no_recommendations = recommendations.empty

    return render_template('index.html', recommendations=recommendations_list, user_id=user_id, genres=genres.columns.tolist(), no_recommendations=no_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
