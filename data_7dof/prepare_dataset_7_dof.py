import matplotlib
import matplotlib.pyplot as plt

from FK import *

matplotlib.use('Qt5Agg')


def visualize_end_x():
    o = np.genfromtxt('data_7dof/JointData.txt')  # (N,7)

    trajectory = np.zeros((o.shape[0], 3))  # not homogeneous

    for row in range(o.shape[0]):
        g, x = fk(xis, o[row], g_0)
        end_point = g @ (np.asarray([[0, 0, 0, 1]]).reshape((4, 1)))
        end_point = end_point[0:3] / end_point[3]
        trajectory[row] = end_point.flatten()

    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')
    ax3D.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Z')
    plt.show()


def prepare_data(path, o, noise=None):
    i_without_noise = np.zeros((o.shape[0], 7))

    for row in range(o.shape[0]):
        g, x = fk(xis, o[row], g_0)
        i_without_noise[row] = x

    if noise is None:
        noise = np.zeros_like(i_without_noise)

    i = i_without_noise + noise

    np.savez_compressed(path, o=o, i=i,
                        i_without_noise=i_without_noise)


def prepare_data_cartesian(path, o, noise=None):
    i_without_noise = np.zeros((o.shape[0], 3))

    for row in range(o.shape[0]):
        g, x = fk(xis, o[row], g_0)
        i_without_noise[row] = x[:3]

    if noise is None:
        noise = np.zeros_like(i_without_noise)

    i = i_without_noise + noise

    np.savez_compressed(path, o=o, i=i,
                        i_without_noise=i_without_noise)


if __name__ == '__main__':
    g_0 = np.asarray([
        [0, -1, 0, 0.61],
        [1, 0, 0, 0.72],
        [0, 0, 1, 2.376],
        [0, 0, 0, 1]
    ])
    n_joints = 7
    xis = np.zeros((6, n_joints))
    xis[0:3, 0] = xis[0:3, 2] = xis[0:3, 4] = xis[0:3, 6] = [0.72, -0.61, 0]

    xis[0:3, 1] = [0, -1.346, 0.72]
    xis[0:3, 3] = [0, -1.896, 0.765]
    xis[0:3, 5] = [0, -2.196, 0.72]

    xis[3:, 0] = xis[3:, 2] = xis[3:, 4] = xis[3:, 6] = [0, 0, 1]
    xis[3:, 1] = xis[3:, 3] = xis[3:, 5] = [-1, 0, 0]
    rng = np.random.default_rng()

    # prepare_data('data_7dof/data_txt.npz', o=np.genfromtxt('data_7dof/JointData.txt'))
    # o = rng.random(size=(5000, 7)) * np.pi * 2 - np.pi  # -pi to pi
    # prepare_data('data_7dof/data_random_without_noise.npz', o=o)
    # noise_rng = np.random.default_rng()
    # noise = noise_rng.normal(scale=np.pi * 0.1, size=o.shape)
    # prepare_data('data_7dof/data_random_with_noise.npz', o=o, noise=noise)

    o = rng.random(size=(5000, 7)) * np.pi * 2 - np.pi  # -pi to pi
    prepare_data_cartesian('data_cart_without_noise.npz', o=o)
