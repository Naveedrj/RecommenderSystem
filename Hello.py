import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies_df = pd.read_csv('movies.csv',usecols=['movieId','title'],
	dtype={'movieId': 'int32', 'title': 'str'})

rating_df=pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df = pd.merge(rating_df,movies_df,on='movieId')


combine_movie_rating = df.dropna(axis = 0)
movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )

Movie_dataframe = pd.merge(df,movie_ratingCount,on='title')

popularity_threshold = 50
rating_popular_movie= Movie_dataframe.query('totalRatingCount >= @popularity_threshold')

movie_features_df=rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)


movie_features_df_matrix = csr_matrix(movie_features_df.values)


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)

query_index = np.random.choice(movie_features_df.shape[0])

distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 4)


for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Movie name is : {0}\n\nRecommendations are \n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}'.format(i, movie_features_df.index[indices.flatten()[i]] ))