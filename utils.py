# Variational Wasserstein Clustering (vwc)
# Author: Liang Mi <icemiliang@gmail.com>
# Date: May 13th 2019

import numpy as np
from PIL import Image


color_blue = [0.12, 0.56, 1]
color_light_blue = [0.5, 0.855, 1]
color_dark_blue = [0.05, 0.28, 0.63]

color_red = [0.8, 0.22, 0]
color_light_red = [1.0, 0.54, 0.5]

color_light_grey = [0.7, 0.7, 0.7]


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", ( w ,h ), buf.tostring())
