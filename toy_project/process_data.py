import os
import pickle
import pandas as pd
import re
import numpy as np
from builder import PandasGraphBuilder
import argparse
from data_utils import *


# https://lsjsj92.tistory.com/569?category=853217

def movielens_to_graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path

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

    # Filter the users and items that never appear in the rating table
    distinct_users_in_ratings = ratings['user_id'].unique()
    distinct_movies_in_ratings = ratings['movie_id'].unique()
    users = users[users['user_id'].isin(distinct_users_in_ratings)]
    movies = movies[movies['movie_id'].isin(distinct_movies_in_ratings)]

    # Group the movie features into genres (a vector), year (a category), title (a string)
    genre_columns = movies.columns.drop(['movie_id', 'title', 'year'])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')
    movies_categorical = movies.drop('title', axis=1)

    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')
    graph_builder.add_entities(movies_categorical, 'movie_id', 'movie')
    graph_builder.add_binary_relations(ratings, 'user_id', 'movie_id', 'watched')
    graph_builder.add_binary_relations(ratings, 'movie_id', 'user_id', 'watched-by')

    g = graph_builder.build()

    # Assign features.
    # Note that variable-sized features such as texts or images are handled elsewhere.
    g.nodes['user'].data['gender'] = torch.LongTensor(users['gender'].cat.codes.values)
    g.nodes['user'].data['age'] = torch.LongTensor(users['age'].cat.codes.values)
    g.nodes['user'].data['occupation'] = torch.LongTensor(users['occupation'].cat.codes.values)
    g.nodes['user'].data['zip'] = torch.LongTensor(users['zip'].cat.codes.values)

    g.nodes['movie'].data['year'] = torch.LongTensor(movies['year'].cat.codes.values)
    g.nodes['movie'].data['genre'] = torch.FloatTensor(movies[genre_columns].values)

    g.edges['watched'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['watched'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)
    g.edges['watched-by'].data['rating'] = torch.LongTensor(ratings['rating'].values)
    g.edges['watched-by'].data['timestamp'] = torch.LongTensor(ratings['timestamp'].values)


    # Train-validation-test split
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')

    # Build the graph with training interactions only.
    train_g = build_train_graph(g, train_indices, 'user', 'movie', 'watched', 'watched-by')
    assert train_g.out_degrees(etype='watched').min() > 0

    # Build the user-item sparse matrix for validation and test set.
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'movie', 'watched')

    ## Build title set

    movie_textual_dataset = {'title': movies['title'].values}

    # The model should build their own vocabulary and process the texts.  Here is one example
    # of using torchtext to pad and numericalize a batch of strings.
    #     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
    #     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
    #     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
    #     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
    #     token_ids, lengths = field.process([examples[0].title, examples[1].title])

    ## Dump the graph and the datasets

    dataset = {
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': movie_textual_dataset,
        'item-images': None,
        'user-type': 'user',
        'item-type': 'movie',
        'user-to-item-type': 'watched',
        'item-to-user-type': 'watched-by',
        'timestamp-edge-column': 'timestamp'}

    print(dataset)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

# 그래프 구조 아닐 때 

def movielens_to_matrix():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--directory', type=str)
    # args = parser.parse_args()
    # directory = args.directory
    directory = './ml-1m'

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
    # 어떤 비율로 표본 추출을 하고싶은지 df.sample에서 frac을 0~1 사이로 설정
    df = ratings.sample(frac=1).reset_index(drop=True)
    sparse_matrix = df.groupby('movie_id').apply(lambda x : pd.Series(x['rating'].values, index=x['user_id'])).unstack()
    val_set = []
    test_set = []
    # to_numpy()는 벡터나 행렬을 ndarray형태로 바꿔줌
    # nonzero()는 0이 아닌 값들의 인덱스를 알려줌. idx는 row index, jdx는 column index라 생각하면 됨
    idx, jdx = sparse_matrix.fillna(0).to_numpy().nonzero()
    indice = list(zip(idx, jdx))
    np.random.shuffle(indice)
    # train, val, test를 80/10/10의 비율로 나눔
    val_sparse_matrix = sparse_matrix.copy()
    test_sparse_matrix = sparse_matrix.copy()
    #val_sparse_matrix = val_sparse_matrix.replace([1,2,3,4,5], 0).fillna(0)
    test_sparse_matrix = test_sparse_matrix.replace([1,2,3,4,5], 0).fillna(0)

    # for i,j in indice[:ratings.shape[0] // 10]:
    #     val_set.append((i, j, sparse_matrix.iloc[i, j]))
    #     val_sparse_matrix.iloc[i,j] = sparse_matrix.iloc[i,j]
    #     sparse_matrix.iloc[i, j] = 0

    for i,j in indice[: ratings.shape[0] // 5]:
        test_set.append((i,j,sparse_matrix.iloc[i, j]))
        test_sparse_matrix.iloc[i,j] = sparse_matrix.iloc[i,j]
        sparse_matrix.iloc[i, j] = 0

    # return ratings, sparse_matrix, val_sparse_matrix, test_sparse_matrix, val_set, test_set
    return ratings, sparse_matrix, test_sparse_matrix, test_set

if __name__=='__main__':
    # 위에서 만든 여러 데이터 전처리 여기서 함수 호출하여 사용
    # graph : python process_data.py ./ml-1m ./movielens_graph.pkl 
    # matrix : python process_data.py ./ml-1m
    movielens_to_graph()
    # sparse_matrix, val_set, test_set = movielens_to_matrix()





    

    