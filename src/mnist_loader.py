import gzip
import struct
from typing import List, Tuple

import numpy as np


def one_hot_encode(n: int, idx: int) -> List[int]:
    hot = np.zeros((n, 1), dtype="float32")
    hot[idx] = 1
    return hot


def load_images_and_labels(imgs_path: str, labels_path: str) -> List[Tuple[np.ndarray[float]]]:
    """
    Load images and labels using the `IDX` file format specified in http://yann.lecun.com/exdb/mnist/

    Label files have the following format:
    ```
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

    The labels values are 0 to 9. 
    ```

    While image files have the following format:
    ```
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel

    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
    ```
    """

    with gzip.open(imgs_path) as f_imgs, gzip.open(labels_path) as f_labels:
        [magic] = struct.unpack(">I", f_imgs.read(4))
        assert magic == 2051, "Invalid images magic byte, file could be corrupted"
        [magic] = struct.unpack(">I", f_labels.read(4))
        assert magic == 2049, "Invalid labels magic byte, file could be corrupted"

        [n_labels] = struct.unpack(">I", f_labels.read(4))
        n_images, n_rows, n_cols = struct.unpack(">III", f_imgs.read(12))
        assert n_labels == n_images, "Number of labels does not equal number of images"

        images = []
        n_pixels = n_rows * n_cols
        for _ in range(n_images):
            pixels = struct.unpack(f">{n_pixels}B", f_imgs.read(n_pixels))
            pixels = np.reshape(pixels, (n_pixels, 1)) / 255
            images.append(pixels)
        labels = struct.unpack(f">{n_labels}B", f_labels.read(n_labels))

    return [(img, one_hot_encode(10, label)) for img, label in zip(images, labels)]
