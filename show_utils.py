import numpy as np
import cv2
import matplotlib.pyplot as plt


def _read_image_if_need(*args):
    """
    for each item read image if it is path

    :param args: list: images or image paths
    :return: list: images
    """
    images = []
    for image in args:
        if type(image) == str:
            image = cv2.imread(image)
        images.append(image)
    return images


def show_image(image, figsize=(20, 20), title=None, cmap=None):
    """
    show single image

    :param image: np.array: image
    :param figsize: (int, int): figure size
    :param cmap: str: colormap name, for example "Greys_r"
    :return: None
    """
    image = _read_image_if_need(image)[0]
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(image, cmap)
    plt.show()


def subplot_images(images, n_rows, n_columns, figsize=(20, 20), titles=None, cmap=None):
    """
    show several images

    :param images: list(np.array): list of images
    :param n_rows: int: row number
    :param n_columns: int: column number
    :param figsize: (int, int): figure size
    :param cmap: str: colormap name, for example "Greys_r"
    :return: None
    """
    assert len(images) <= n_rows * n_columns
    if titles is None or type(titles) == str:
        titles = [titles] * len(images)
    plt.figure(figsize=figsize)
    for i, image in enumerate(images, start=1):
        plt.subplot(n_rows, n_columns, i)
        plt.title(titles[i-1])
        plt.imshow(image, cmap)
    plt.show()