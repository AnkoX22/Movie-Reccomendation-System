import numpy as np
import pandas as pd
import data_loader

def np_cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def manual_cosine_similarity(x, y):
    if len(x) != len(y):
        raise ValueError("Vectors must be the same length")

    dot_product = 0
    x_squared = 0
    y_squared = 0

    for i in range(len(x)):
        dot_product += x[i] * y[i]
        x_squared += x[i] * x[i]
        y_squared += y[i] * y[i]

    denominator = np.sqrt(x_squared * y_squared)

    if denominator == 0:
        return 0

    return dot_product / denominator

def np_pearson_correlation(x, y):
    x_centered = x - x.mean()
    y_centered = x - y.mean()

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2))*np.sqrt(np.sum(y_centered**2))

    if denominator == 0:
        return 0
    return numerator/denominator

def manual_pearson_correlation(x, y):

    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("Vectors must be the same length")

    if len(x) == 0:
        return 0

    x_total = 0
    y_total = 0

    for i in range(len(x)):
        x_total += x[i]
        y_total += y[i]

    x_average = x_total / len(x)
    y_average = y_total / len(y)

    x_centered = x - x_average
    y_centered = y - y_average

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2))*np.sqrt(np.sum(y_centered**2))

    return numerator / denominator

def center_array(x):
    """Scale the array"""

    x_mean = np.mean(x)
    if x_mean == 0:
        return x

    return x - x_mean

def root_mean_squared_error(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def user_based_cf_merge_movies(user_id, array):
    """
    Calculate the users similarity according to cosine similarity & pearson correlation
    using only the movies that both of the users have rated
    """

    user_ratings = array[ array['userId'] == user_id]
    cosine_similarity = []
    pearson_correlation = []


    for other_id in array['userId'].unique():
        if other_id == user_id:
            continue


        other_user_rating = array[array['userId'] == other_id]
        common = pd.merge(user_ratings, other_user_rating, on='movieId', suffixes=('_u1', '_u2'))

        if len(common) == 0:
            continue

        v1 = common['rating_u1'].values
        v2 = common['rating_u2'].values
        cosine_similarity.append([int(other_id), float(np_cosine_similarity(v1, v2))])
        pearson_correlation.append([int(other_id), float(np_pearson_correlation(v1, v2))])

    return pd.DataFrame(cosine_similarity, columns=['userId', 'cosine']).sort_values('userId').reset_index(drop=True), pd.DataFrame(pearson_correlation, columns=['userId', 'pearson']).sort_values('userId').reset_index(drop=True)


def user_based_cf_union_movies(user_id, ratings_matrix):
    """
    Compute cosine & Pearson similarity between one user
    and all others based on the user–item rating matrix.
    """
    target_ratings = ratings_matrix.loc[user_id].values
    cosine_similarities = []
    pearson_correlations = []

    for other_id in ratings_matrix.index:
        if other_id == user_id:
            continue  # skip self

        other_ratings = ratings_matrix.loc[other_id].values

        cos_sim = np_cosine_similarity(target_ratings, other_ratings)
        pear_corr = np_pearson_correlation(target_ratings, other_ratings)

        cosine_similarities.append([other_id, float(cos_sim)])
        pearson_correlations.append([other_id, float(pear_corr)])

    return pd.DataFrame(cosine_similarities, columns=['userId', 'cosine']).sort_values('userId').reset_index(drop=True), pd.DataFrame(pearson_correlations, columns=['userId', 'pearson']).sort_values('userId').reset_index(drop=True)

def item_based_cf_merge_users(movie_id, array):
    """
    Compute the item ( movies ) similarity using cosine similarity
    & pearson correlation
    """

    movie_ratings = array[ array['movieId'] == movie_id]
    cosine_similarity = []
    pearson_correlation = []

    for other_id in array['movieId'].unique():
        if other_id == movie_id:
            continue

        other_ratings = array[array['movieId'] == other_id]
        common = pd.merge(movie_ratings, other_ratings, on='userId', suffixes=('_m1', '_m2'))

        if len(common) == 0:
            continue

        m1 = common['rating_m1'].values
        m2 = common['rating_m2'].values

        cosine_similarity.append([int(other_id), float(np_cosine_similarity(m1, m2))])
        pearson_correlation.append([int(other_id), float(np_pearson_correlation(m1, m2))])

    return pd.DataFrame(cosine_similarity, columns=['movieId', 'cosine']).sort_values('movieId').reset_index(drop=True), pd.DataFrame(pearson_correlation, columns=['movieId', 'pearson']).sort_values('movieId').reset_index(drop=True)

def item_based_cf_union_users(movie_id, array):
    """
    find the cosine similarity & pearson correlation between movies
    this time with the union of ratings for the 2 movies being compared every time
    """

    target_rating = array.loc[movie_id].values
    cosine_similarities = []
    pearson_correlation = []

    for other_id in array['movieId'].unique():

        if other_id == movie_id:
            continue

        other_ratings = array.loc[other_id].values
        cos_sim = np_cosine_similarity(target_rating, other_ratings)
        pear_corr = np_pearson_correlation(other_ratings, target_rating)

        cosine_similarities.append([other_id, float(cos_sim)])
        pearson_correlation.append([other_id, float(pear_corr)])

    return pd.DataFrame(cosine_similarities, columns=['movieId', 'cosine']).sort_values('movieId').reset_index(drop=True), pd.DataFrame(pearson_correlation, columns=['movieId', 'pearson']).sort_values('movieId').reset_index(drop=True)

if __name__ == "__main__":
    ratings, movies, users, genres, occupation = data_loader.load_data()
    user_based_array = ratings[['userId', 'movieId', 'rating']]
    print(f"user_based_array: {user_based_array}")
    print(users['userId'][0])


    print(ratings[ ratings['userId'] == 1 ].reset_index(drop=True))

    print("\n")
    print("--------------------------------------------------------------------------------------------")
    print("user_based_cf_union_movies(1, user_based_array)")
    cosine_similarity, pearson_correlation = user_based_cf_merge_movies(1, user_based_array)
    print(cosine_similarity)
    print(pearson_correlation)

    print("\n")
    print("--------------------------------------------------------------------------------------------")
    print("user_based_cf_merge_movies(1, user_based_array)")
    ratings_matrix = user_based_array.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    cosine_similarity2, pearson_correlation2 = user_based_cf_union_movies(1, ratings_matrix)
    print(cosine_similarity2)
    print(pearson_correlation2)

    print("\n")
    print("--------------------------------------------------------------------------------------------")
    print("item_based_cf_union_movies(1, movie_based_array)")
    cosine_similarity3, pearson_correlation3 = item_based_cf_union_users(1, user_based_array)
    print(cosine_similarity3)
    print(pearson_correlation3)

    print("\n")
    print("--------------------------------------------------------------------------------------------")
    print("item_based_cf_merge_movies(1, movie_based_array)")
    cosine_similarity4, pearson_correlation4 = item_based_cf_merge_users(1, user_based_array)
    print(cosine_similarity4)
    print(pearson_correlation4)