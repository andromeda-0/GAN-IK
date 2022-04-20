import matplotlib
import matplotlib.pyplot as plt

from FK import *

matplotlib.use('Qt5Agg')


def visualize_end_x():
    angles = np.genfromtxt('JointData.txt')  # (N,7)

    trajectory = np.zeros((angles.shape[0], 3))  # not homogeneous

    for row in range(angles.shape[0]):
        g, x = fk(xis, angles[row], g_0)
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


def prepare_data_from_txt():
    angles = np.genfromtxt('JointData.txt')  # (N,7)

    configurations = np.zeros((angles.shape[0], 7))

    for row in range(angles.shape[0]):
        g, x = fk(xis, angles[row], g_0)
        configurations[row] = x

    np.savez_compressed('data_txt.npz', angles=angles, configurations=configurations)


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

    prepare_data_from_txt()
