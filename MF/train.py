import argparse
from pprint import pprint
import pickle
import torch
from process_data import movielens_to_matrix
from models.matrix_factorization import MF
from models.pinsage import train
from metrics import precision_at_k, recall_at_k

def define_argparser():
    # 학습 시 필요한 인자들 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, help='latent facotr size')
    parser.add_argument('--k', type=int, default=10, help='number of recommendations')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.01, help='regularization parameter')
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')  # 'cpu' or 'cuda:N'
    parser.add_argument('--batches-per-epoch', type=int, default=10000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--MF', action='store_true', help='Use MF algorithm')
    parser.add_argument('--pinsage', action='store_true', help='Use pinsage algorithm')
    args = parser.parse_args()
    return args

def main(config):
    pprint(vars(config))
    # MF를 위한 sparse matrix 불러오기
    # ratings, sparse_matrix, val_sparse_matrix, test_sparse_matrix, val_set, test_set = movielens_to_matrix()
    

    # 실행할 model 정하고 객체 만들기
    # python train.py --MF --hidden_dim 300 --k 10 --lr 0.01 --beta 0.001 --epoch 5
    if config.MF:
        ratings, sparse_matrix, test_sparse_matrix, test_set = movielens_to_matrix()
        print('Dataset shape : ', ratings.shape)
        print('Train sparse Matrix shape : ', sparse_matrix.T.shape)
        # print('Validation set length : ', len(val_set))
        print('Test set length : ', len(test_set))
        trainer = MF(
            sparse_matrix,
            config.hidden_dim,
            config.k,
            config.lr,
            config.beta,
            config.epochs
        )
        trainer.train()
        print('train RMSE', trainer.evaluate())
        print('recommendations : ', trainer.get_recommendation().shape)
        print('precision@k : ', precision_at_k(trainer.get_recommendation(), test_sparse_matrix))
        print('recall@k : ', recall_at_k(trainer.get_recommendation(), test_sparse_matrix))
        # print('validation RMSE', trainer.val_evaluate(val_set))
        print('test RMSE', trainer.test_evaluate(test_set))

    # python train.py --pinsage --dataset-path data.pkl --epochs 300 --num-workers 16 --device cuda:0 --hidden-dims 64 --batches-per-epoch 10000
    elif config.pinsage:
        with open(config.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        train(dataset, config)
    # elif:
        # TODO
    
    else:
        raise RuntimeError('Algorithm not selected')

    

if __name__ == '__main__':
    config = define_argparser()
    main(config)

