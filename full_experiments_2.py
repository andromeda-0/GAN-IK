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
    pool = multiprocessing.Pool(1)

    arg = Params(learning='ssGAN', z_method='batch', generator='Generator_3',
                 data_path='data_3dof/SomeVars3.mat',
                 dataset='DynamicsSet',
                 gpu_id=1)
    pool.apply_async(test, (arg,), error_callback=error_callback)

    pool.close()
    pool.join()
