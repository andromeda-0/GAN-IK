import multiprocessing

# noinspection PyUnresolvedReferences
from models import *
# noinspection PyUnresolvedReferences
from learning import *


def test(args):
    learning = eval(args.learning + '(args)')
    learning.train()
    print(args)


if __name__ == '__main__':
    pool = multiprocessing.Pool(3)
    for learning in ['GAN', 'wGAN', 'ssGAN']:
        for z_method in ['add', 'mini_batch', 'batch']:
            for generator in ['Generator_1', 'Generator_2', 'Generator_3']:
                arg_cart = Params(learning=learning, z_method=z_method, generator=generator,
                                  data_path='data_7dof/data_cart_without_noise.npz',
                                  gpu_id=1)
                arg_k = Params(learning=learning, z_method=z_method, generator=generator,
                               data_path='data_7dof/data_random_without_noise.npz',
                               gpu_id=1)
                t_1 = pool.apply_async(test, (arg_cart,))
                t_2 = pool.apply_async(test, (arg_k,))

    pool.close()
    pool.join()
