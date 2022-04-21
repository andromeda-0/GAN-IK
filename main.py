import argparse

# noinspection PyUnresolvedReferences
from models import *
# noinspection PyUnresolvedReferences
from learning import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument('--data_path', default='data_7dof/data_txt.npz')
    parser.add_argument('--generator', default='cGenerator_0')
    parser.add_argument('--discriminator', default='cDiscriminator')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--learning', default='cGAN')
    parser.add_argument('--n_critic', default=5, type=int)
    args = parser.parse_args()

    learning = eval(args.learning + '(args)')
    learning.train()
