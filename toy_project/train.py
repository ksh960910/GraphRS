import argparse
from pprint import pprint
from process_data import movielens_to_matrix, movielens_to_graph
from models.matrix_factorization import MF

def define_argparser():
    # 학습 시 필요한 인자들 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, help='latent facotr size')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.01, help='regularization parameter')
    parser.add_argument('--MF', action='store_true', help='Use MF algorithm')
    args = parser.parse_args()
    return args

def main(config):
    pprint(vars(config))
    # MF를 위한 sparse matrix 불러오기
    ratings, sparse_matrix, val_set, test_set = movielens_to_matrix()
    print('Dataset shape : ', ratings.shape)
    print('Sparse Matrix shape : ', sparse_matrix.shape)
    print('Validation set length : ', len(val_set))
    print('Test set length : ', len(test_set))

    # 실행할 model 정하고 객체 만들기
    if config.MF:
        trainer = MF(
            sparse_matrix,
            config.k,
            config.lr,
            config.beta,
            config.epochs
        )
    # elif:
        # TODO
    else:
        raise RuntimeError('Algorithm not selected')

    trainer.train()
    print('train RMSE', trainer.evaluate())
    print('validation RMSE', trainer.val_evaluate(val_set))
    print('test RMSE', trainer.test_evaluate(test_set))

if __name__ == '__main__':
    config = define_argparser()
    main(config)

