import numpy as np


def vee(vector_hat: np.ndarray) -> np.ndarray:
    """

    :param vector_hat: (3,3) or (4,4)
    :return: (3,1) or (6,1)
    """

    if vector_hat.shape == (3, 3):
        vector = np.vstack([vector_hat[2, 1], vector_hat[0, 2], vector_hat[1, 0]]).reshape((-1, 1))
    elif vector_hat.shape == (4, 4):
        vector = np.vstack([vector_hat[0:3, 3], vee(vector_hat[0:3, 0:3])]).reshape((-1, 1))
    else:
        raise ValueError()

    return vector


def wedge(vector: np.ndarray) -> np.ndarray:
    """

    :param vector: (3,1) or (6,1)
    :return: (3,3) or (4,4)
    """
    if vector.shape[0] == 3:
        wedged = np.asarray([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    elif vector.shape[0] == 6:
        wedged = np.zeros((4, 4))
        wedged[0:3, 0:3] = wedge(vector[3:])
        wedged[0:3, 3] = vector[0:3]
    else:
        raise ValueError()
    return wedged


def exp_omegah_theta(omega_h, theta: float):
    if omega_h.shape != (3, 3):
        raise ValueError()

    return np.eye(3) + np.sin(theta) * omega_h + (1 - np.cos(theta)) * omega_h @ omega_h


def exp_xih_theta(xih: np.ndarray, theta: float) -> np.ndarray:
    """

    :param xih: (4,4)
    :param theta:
    :return:
    """

    omega_h = xih[0:3, 0:3]
    v = (xih[0:3, 3]).reshape((-1, 1))
    omega = vee(omega_h)
    e_o_t = exp_omegah_theta(omega_h, theta)

    res = np.zeros((4, 4))
    res[0:3, 0:3] = e_o_t
    res[0:3, 3] = ((np.eye(3) - e_o_t) @ omega_h @ v
                   + omega @ omega.transpose() @ v * theta).flatten()
    res[3, 3] = 1

    return res
