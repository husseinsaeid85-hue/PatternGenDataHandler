"""Synthetic 2D test patterns generated with NumPy.

Each pattern is a small class with the same three-part interface:

    ``__init__``  store the geometry parameters
    ``draw()``    build the pattern into ``self.output`` and return a copy
    ``show()``    render the pattern with matplotlib

The patterns are deterministic and dependency-free (NumPy only), which makes
them useful as fixed inputs when sanity-checking a neural network's data path.
"""

import numpy as np
import matplotlib.pyplot as plt


class Checker:
    """Black-and-white checkerboard pattern.

    The pattern is built from a single 2x2 tile block that is tiled across the
    full canvas, so the resolution must be an integer multiple of twice the
    tile size (``resolution % (2 * tile_size) == 0``). The top-left tile is
    black.

    Args:
        resolution: Side length of the square output image, in pixels.
        tile_size: Side length of a single checker tile, in pixels.

    Attributes:
        output: ``(resolution, resolution)`` float array of 0.0 / 1.0 values.
    """

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution))

    def draw(self):
        """Build the checkerboard and return a copy of it.

        Returns:
            A ``(resolution, resolution)`` array. If the resolution is not an
            integer multiple of twice the tile size the pattern cannot be tiled
            evenly, a message is printed and the zero-initialised output is
            returned unchanged.
        """
        if self.resolution % (2 * self.tile_size) != 0:
            print("Cannot draw the checkerboard!")

        else:
            blk = np.zeros((self.tile_size, self.tile_size))
            wht = np.ones((self.tile_size, self.tile_size))
            merge = np.concatenate((blk, wht), axis=1)
            merge = np.concatenate((merge, np.flip(merge, axis=1)), axis=0)
            rep = int((self.resolution / self.tile_size) / 2)
            self.output = np.tile(merge, (rep, rep))

        return self.output.copy()

    def show(self):
        """Draw the pattern and display it in greyscale."""
        plt.imshow(self.draw(), cmap='gray')
        plt.axis('off')
        plt.show()


class Circle:
    """Binary filled-circle (disc) pattern.

    Args:
        resolution: Side length of the square output image, in pixels.
        radius: Radius of the disc, in pixels.
        centers: ``(x, y)`` pixel coordinates of the disc centre.

    Attributes:
        output: ``(resolution, resolution)`` boolean array, ``True`` inside the
            disc.
    """

    def __init__(self, resolution, radius, centers):
        self.resolution = resolution
        self.radius = radius
        self.centers = tuple(centers)
        self.output = np.zeros((resolution, resolution))

    def draw(self):
        """Build the disc mask and return a copy of it.

        The mask is evaluated on a coordinate grid, so every pixel is tested
        against the circle equation at once rather than in a Python loop.

        Returns:
            A ``(resolution, resolution)`` boolean array.
        """
        x_axis = np.arange(self.resolution)
        y_axis = np.arange(self.resolution)
        xx, yy = np.meshgrid(x_axis, y_axis)
        x_centers, y_centers = self.centers
        self.output = (((xx - x_centers) ** 2) + ((yy - y_centers) ** 2) <= self.radius ** 2)
        return self.output.copy()

    def show(self):
        """Draw the pattern and display it in greyscale."""
        plt.imshow(self.draw(), cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    """RGB colour-spectrum pattern.

    Each channel is a linear ramp: red increases left to right, green increases
    top to bottom, and blue decreases left to right. The result is a continuous
    colour gradient across the canvas.

    Args:
        resolution: Side length of the square output image, in pixels.

    Attributes:
        output: ``(resolution, resolution, 3)`` float array with values in
            ``[0, 1]``. Populated during construction.
    """

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = self.draw()

    def draw(self):
        """Build the spectrum and return it.

        Returns:
            A ``(resolution, resolution, 3)`` float array with values in
            ``[0, 1]``.
        """
        resolution = self.resolution
        img = np.zeros([resolution, resolution, 3])
        img[:, :, 0] = np.linspace(0, 1, resolution)
        img[:, :, 1] = np.linspace(0, 1, resolution).reshape(resolution, 1)
        img[:, :, 2] = np.linspace(1, 0, resolution)

        return img.copy()

    def show(self):
        """Display the spectrum built during construction."""
        plt.imshow(self.output)
        plt.show()
