import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run BGCF')
    parser.add_argument('--path', default='Data/', help = 'Input data path')
    parser.add_argument('--weights_path', nargs='?', default='ml-1m/weight', help='Store model path.')

    parser.add_argument('--dataset', default='ml-1m', help = 'Choose a dataset from {movielens, gowalla, amazon-book')

    parser.add_argument('--epsilon' , type=float, default=1e-8, help = 'Node copying probability')
    parser.add_argument('--verbose' , type=int, default=1, help = 'Interval of evaluation')
    parser.add_argument('--epoch', type=int, default=600, help = 'Number of epoch')
    parser.add_argument('--embed_size', type=int, default=64, help = 'Embedding size')
    parser.add_argument('--layer_size', default=64, help='Output size of layer')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--sample_num', type=int, default=1, help='Number of using generated graphs')

    parser.add_argument('--l2', default=0.03, help='L2')
    parser.add_argument('--regs', default=0.003, help = 'Regularizations')
    parser.add_argument('--lr', type=float, default=0.001, help = 'Learning rate')
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True, help='Use Cuda')

    parser.add_argument('--node_dropout_flag', type=int, default=1, help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', default='[0.2]', 
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout')
    parser.add_argument('--mess_dropout', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1 : no dropout')
    
    parser.add_argument('--Ks', default='[20,40,60]', help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0, help='0: Disable model saver, 1: Activate model saver')

    return parser.parse_args()
