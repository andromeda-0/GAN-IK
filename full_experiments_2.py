import multiprocessing
import traceback

# noinspection PyUnresolvedReferences
from learning import *
# noinspection PyUnresolvedReferences
from models import *


def error_callback(e):
    traceback.print_exception(type(e), e, e.__traceback__)


def test(args):
    learning = eval(args.learning + '(args)')
    learning.train()
    print(args, flush=True)


if __name__ == '__main__':
    pool = multiprocessing.Pool(4)
    for learning in ['GAN', 'wGAN', 'ssGAN']:
        for z_method in ['add', 'minibatch', 'batch']:
            for generator in ['Generator_1', 'Generator_2', 'Generator_3']:
                arg_cart = Params(learning=learning, z_method=z_method, generator=generator,
                                  data_path='data_7dof/data_cart_without_noise.npz',
                                  gpu_id=0)
                arg_k = Params(learning=learning, z_method=z_method, generator=generator,
                               data_path='data_7dof/data_random_without_noise.npz',
                               gpu_id=0)
                t_1 = pool.apply_async(test, (arg_cart,), error_callback=error_callback)
                t_2 = pool.apply_async(test, (arg_k,), error_callback=error_callback)

    pool.close()
    pool.join()
