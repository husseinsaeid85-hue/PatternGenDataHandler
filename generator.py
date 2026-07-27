"""Batch loader for a directory of ``.npy`` images with JSON labels.

``ImageGenerator`` is the data-feeding half of this repository: it reads images
off disk once, then hands out fixed-size batches of ``(images, labels)`` on
each call to :meth:`ImageGenerator.next`, applying optional resizing, shuffling
and augmentation on the way out. It keeps track of how many full passes over
the dataset have been made so training loops can query the current epoch.
"""

import os.path
import json
import glob
import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator:
    """Yields batches of images and labels from a directory of ``.npy`` files.

    Images are expected to be individual ``.npy`` arrays named by their integer
    index (``0.npy``, ``1.npy``, ...) inside ``file_path``. ``label_path``
    points at a JSON file mapping those same indices, as strings, to integer
    class ids -- for example ``{"0": 6, "1": 9, ...}``.

    All images are loaded into memory during construction, which is fine for
    the small sample dataset shipped with this repository.

    Sampling is without replacement within an epoch: each call to :meth:`next`
    draws a fresh batch from the indices not yet used, and once the pool is
    exhausted the epoch counter advances and the pool is refilled.

    Args:
        file_path: Directory containing the ``.npy`` image files.
        label_path: Path to the JSON file holding the index-to-class mapping.
        batch_size: Number of images returned per call to :meth:`next`.
        image_size: Target shape as ``[height, width, channels]``; channels of
            3 means RGB.
        rotation: If ``True``, randomly rotate images by 90, 180 or 270 degrees.
        mirroring: If ``True``, randomly flip images horizontally or vertically.
        shuffle: If ``True``, shuffle the dataset order on each call.

    Attributes:
        image_list: All images loaded from ``file_path``, in index order.
        labels: The parsed contents of ``label_path``.
        epochs: Number of complete passes made over the dataset so far.
        class_dict: Mapping from integer class id to human-readable class name.
    """

    def __init__(self, file_path, label_path, batch_size, image_size,
                 rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size   # [height, width, channel] ex: ch=3 -> RGB image
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.image_list = []
        self.batch = []
        self.epochs = 0
        self.epochs_list = []

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        # Load the image files, ordered by their numeric file name so that an
        # image's position in image_list matches its key in the label file.
        files = sorted(glob.glob(os.path.join(self.file_path, "*.npy")),
                       key=lambda x: int(os.path.basename(x).split(".")[0]))
        self.image_list = [np.load(f) for f in files]

        # Load the labels
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        # Pool of sample indices still unused in the current epoch.
        self.n_sample = np.arange(len(self.image_list))

    def next(self):
        """Return the next batch of images and labels.

        Resizes any image that does not already match ``image_size``, optionally
        reshuffles the dataset, then draws ``batch_size`` sample indices from the
        pool of indices not yet used this epoch. When fewer than ``batch_size``
        samples remain, the batch is topped up with samples reused from the start
        of the epoch, the epoch counter is incremented and the pool is refilled.
        Mirroring and rotation, when enabled, are applied per image.

        Returns:
            A ``(images, labels)`` tuple of NumPy arrays, where ``images`` has
            shape ``(batch_size, *image_size)`` and ``labels`` has shape
            ``(batch_size,)``.
        """
        np.random.seed(0)

        # resize option
        self.image_list = [np.resize(img, self.image_size) if img.shape != self.image_size else img for img in
                           self.image_list]

        # shuffle
        if self.shuffle:
            randomize = np.arange(len(self.image_list))
            np.random.shuffle(randomize)
            self.image_list = np.array(self.image_list)[randomize]
            self.labels = {str(i): self.labels[str(randomize[i])] for i in range(len(randomize))}

        if len(self.n_sample) >= self.batch_size:
            t = np.random.choice(self.n_sample, size=self.batch_size, replace=False)
            self.batch.append(t)
            self.n_sample = np.setdiff1d(self.n_sample, t)

        elif len(self.n_sample) == 0:
            self.epochs += 1
            self.batch = []
            self.n_sample = np.arange(len(self.image_list))
            t = np.random.choice(self.n_sample, size=self.batch_size, replace=False)
            self.batch.append(t)
            self.n_sample = np.setdiff1d(self.n_sample, t)
            self.epochs_list.append([self.batch])
        else:
            t = np.concatenate((self.n_sample, self.batch[0][:((self.batch_size) - len(self.n_sample))]))
            self.batch.append(t)
            self.epochs += 1
            self.epochs_list.append([self.batch])
            self.batch = []
            self.n_sample = np.arange(len(self.image_list))

        images = []
        labels = []

        for i in range(self.batch_size):
            im = t[i]

            # mirror function
            if self.mirroring:
                mirror_function = np.random.choice(['lr', 'ud', 'No mirroring'], size=1)
                if mirror_function != 'No mirroring':
                    if mirror_function == 'lr':
                        self.image_list[im] = np.fliplr(self.image_list[im])
                    elif mirror_function == 'ud':
                        self.image_list[im] = np.flipud(self.image_list[im])

            # rotation function
            if self.rotation:
                a = np.random.choice(['0', '90', '180', '270'])
                if a != '0':
                    if a == '90':
                        self.image_list[im] = np.rot90(self.image_list[im])
                    elif a == '180':
                        self.image_list[im] = np.rot90(self.image_list[im], 2)
                    elif a == '270':
                        self.image_list[im] = np.rot90(self.image_list[im], 3)

            images.append(self.image_list[im])
            labels.append(self.labels[str(im)])

        im_arrays = np.array(images)
        lab_arrays = np.array(labels)

        return im_arrays, lab_arrays

    def current_epoch(self):
        """Return the number of complete passes made over the dataset so far."""
        return self.epochs

    def class_name(self, label):
        """Map an integer class id to its human-readable name.

        Args:
            label: Integer class id, as stored in the label file.

        Returns:
            The class name, or ``None`` if the id is not in ``class_dict``.
        """
        return self.class_dict.get(label)

    def show(self):
        """Draw one batch and display it in a grid of labelled subplots."""
        images, labels = self.next()
        fig = plt.figure()
        for i, (image, title) in enumerate(zip(images, labels)):
            fig.add_subplot(3, int(np.ceil(len(images) / float(3))), i + 1)
            plt.title(self.class_name(labels[i]))
            plt.imshow(image)
        plt.show()
