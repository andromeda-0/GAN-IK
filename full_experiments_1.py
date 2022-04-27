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
    pool = multiprocessing.Pool(2)
    for learning in ['ssGAN']:
        for z_method in ['add', 'minibatch', 'batch']:
            for generator in ['Generator_3']:
                arg_d = Params(learning=learning, z_method=z_method, generator=generator,
                               data_path='data_3dof/SomeVars3.mat',
                               dataset='DynamicsSet',
                               gpu_id=1)
                t = pool.apply_async(test, (arg_d,), error_callback=error_callback)

    pool.close()
    pool.join()
