import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run BGCF')
    parser.add_argument('--data_path', default='ml-1m', help = 'Input data path')

    parser.add_argument('--dataset', default='movielens', help = 'Choose a dataset from {movielens, gowalla, amazon-book')

    parser.add_argument('--verbose' , type=int, default=1, help = 'Interval of evaluation')
    parser.add_argument('--epoch', type=int, default=400, help = 'Number of epoch')
    parser.add_argument('--embed_size', type=int, default=64, help = 'Embedding size')
    parser.add_argument('--layer_size', default=64, help='Output size of layer')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')

    parser.add_argument('--regs', default=1e-5, help = 'Regularizations')
    parser.add_argument('--lr', type=float, default=0.0001, help = 'Learning rate')
    
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--node_dropout_flag', type=int, default=1, help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', default='[0.1]', 
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout')
    parser.add_argument('--mess_dropout', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1 : no dropout')
    
    parser.add_argument('--Ks', default='[10,20,40]', help='Output sizes of every layer')

    return parser.parse_args()
