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
    pool = multiprocessing.Pool(2)
    for learning in ['GAN', 'wGAN', 'ssGAN']:
        for z_method in ['add', 'mini_batch', 'batch']:
            for generator in ['Generator_1', 'Generator_2', 'Generator_3']:
                arg_d = Params(learning=learning, z_method=z_method, generator=generator,
                               data_path='data_3dof/SomeVars3.mat',
                               dataset='DynamicsSet',
                               gpu_id=0)
                pool.apply_async(test, (arg_d,))
    pool.close()
    pool.join()
