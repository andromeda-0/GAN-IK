from utils import *
from scipy.spatial.transform.rotation import Rotation


def fk(xi: np.ndarray, thetas, g_0):
    """

    :param xi: (6,N_JOINTS)
    :param thetas:
    :param g_0:
    :return: g (4,4), x = [p,quaternion] (7,)
    """

    n_joints = xi.shape[1]
    assert xi.shape[0] == 6

    g = np.eye(4)
    for i in range(n_joints):
        g = g @ exp_xih_theta(wedge(xi[:, i]), thetas[i])

    g = g @ g_0

    rotation: Rotation = Rotation.from_matrix(g[0:3, 0:3])
    quaternion = rotation.as_quat()
    return g, np.concatenate([g[0:3, 3], quaternion]).flatten()
