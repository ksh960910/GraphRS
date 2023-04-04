Look for the full dataset?
Please visit the [websit](https://snap.stanford.edu/data/loc-gowalla.html).
python main.py --dataset ml-1m --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]