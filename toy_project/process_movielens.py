import os
import pandas as pd
import re

# https://lsjsj92.tistory.com/569?category=853217

class movielens_data:
    def __init__(self, directory):
        self.directory = directory

        users = []
        with open(os.path.join(directory, 'users.dat'), encoding='latin1') as f:
            for l in f:
                id_, gender, age, occupation, zip_ = l.strip().split('::')
                users.append({
                    'user_id' : int(id_),
                    'gender' : gender,
                    'age' : age,
                    'occupation' : occupation,
                    'zip' : zip_,
                })
        users = pd.DataFrame(users).astype('category')

        movies = []
        with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
            for l in f:
                id_, title, genres = l.strip().split('::')
                genres_set = set(genres.split('|'))

                # extract year
                assert re.match(r'.*\([0-9]{4}\)$', title)
                year = title[-5:-1]
                title = title[:-6].strip()

                data = {'movie_id': int(id_), 'title': title, 'year': year}
                for g in genres_set:
                    data[g] = True
                movies.append(data)
        movies = pd.DataFrame(movies).astype({'year': 'category'})

        ratings = []
        with open(os.path.join(directory, 'ratings.dat'), encoding='latin1') as f:
            for l in f:
                user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        ratings = pd.DataFrame(ratings)

        print(movies)

d = movielens_data('ml-1m/')
print(d)