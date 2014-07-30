from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np


class Arrow3D(FancyArrowPatch):
    """
    Draw a vector in a 3d axis. Got this off SE.

    To draw a vector pointing at [1, 1, 1];
    a = Arrow3D(
        [0, 1], [0, 1], [0, 1], mutation_scale=20, lw=1,
        arrowstyle="-|>", color="k")
    ax.add_artist(a)
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def make_cap(r=1.0, cap_angle=0.33 * np.pi, direction=None, usteps=100, vsteps=100):

    if direction is not None:
        assert np.linalg.norm(direction) > 0

    u = np.linspace(0, 2 * np.pi, usteps)
    v = np.linspace(0, cap_angle, vsteps)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    if direction is not None:
        north = np.array([0, 0, 1])
        theta = np.arccos(np.dot(north, direction / np.linalg.norm(direction)))

        if theta > 0:
            axis = np.cross(north, direction)

            R = rotation_matrix(axis, -theta)

            shape = x.shape

            x = np.reshape(x, (1, x.size))
            y = np.reshape(y, (1, y.size))
            z = np.reshape(z, (1, z.size))

            points = np.concatenate((x, y, z), axis=0)
            points = np.dot(R, points)

            x = np.reshape(points[0, :], shape)
            y = np.reshape(points[1, :], shape)
            z = np.reshape(points[2, :], shape)

    return x, y, z

if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    radius = 2.0
    direction = np.array([1, 1, 1])
    cap = make_cap(r=radius, direction=direction)
    ax.plot_surface(*cap, linewidth=0, alpha=0.4, color='b')

    cap = make_cap(r=radius/2.0)
    ax.plot_surface(*cap, linewidth=0, alpha=0.4, color='r')

    cap = make_cap(r=radius)
    ax.plot_surface(*cap, linewidth=0, alpha=0.4, color='g')

    cap = make_cap(r=1.2*radius, cap_angle=0.01 * np.pi, direction=direction)
    ax.plot_surface(*cap, linewidth=0, alpha=1.0, color='k')

    cap = make_cap(r=0.99*radius, cap_angle=np.pi)
    ax.plot_wireframe(
        *cap, cstride=5, rstride=5, linewidth=0.05,
        alpha=0.9, color=(0.1, 0.1, 0.1))

    ax.scatter(0.0, 0.0, 0.0)
    plt.setp(
        ax, xlim=(-radius, radius), ylim=(-radius, radius),
        zlim=(-radius, radius), xlabel='x', ylabel='y', zlabel='z')

    plt.show()
