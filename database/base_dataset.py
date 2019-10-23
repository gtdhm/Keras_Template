# ==============database.base_dataset.py======================
# This module implements a base class for datasets.

# Version: 1.0.0
# Date: 2019.08.07
# ============================================================

import os
import random
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
from util import cal_equal, open_csv_file, open_json_file, progress_bar


###############################################################
# Base Dataset Class
###############################################################
class BaseDataset(object):
    """This class includes base processing for the dataset.
    Inputs:
        cfg: the total options

    Examples:
        <<< train_db = BaseDataset(cfg)
            train_db.load_data(mode='Train', shuffle=True)
    """

    def __init__(self, cfg):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.opts = cfg.opts
        self.image_info = {}
        self.label_info = {}
        self.image_ids = []
        random.seed(cfg.opts.seed)
        # Instance a DataTransforms
        self.transform = DataTransforms(cfg)

    def _add_to_database(self, index, data_set, path):
        # TODO(User) >>> add your own data encoding
        # Add image ids
        self.image_ids.append(index)
        # Add labels info
        self.label_info.update({index: int(data_set[1])})
        # Add images info
        self.image_info.update({index: {"image_name": str(data_set[0]),
                                        "image_path": str(os.path.join(path, data_set[0]))}})
        # TODO(User): End

    def load_data(self, mode, shuffle=False):
        """Load the train or val or test dataset"""
        if mode == 'Train':
            label_name, label_dir, data_dir = [self.opts.train_label, self.cfg.TRAIN_LABEL_DIR, self.cfg.TRAIN_DATA_DIR]
        elif mode == 'Val':
            label_name, label_dir, data_dir = [self.opts.val_label, self.cfg.VAL_LABEL_DIR, self.cfg.VAL_DATA_DIR]
        else:
            label_name, label_dir, data_dir = [self.opts.test_label, self.cfg.TEST_LABEL_DIR, self.cfg.TEST_DATA_DIR]

        if label_name == 'None':
            label_data = []
            test_names = os.listdir(data_dir)
            for name in test_names:
                label_data.append([name, 0])
        else:
            label_data = self._open_data_file(label_name, label_dir)
        if shuffle:
            random.shuffle(label_data)
        for index, data_set in enumerate(label_data):
            if mode == 'Test':
                length = self.opts.num_test if self.opts.num_test < len(label_data) else len(label_data)
                if index + 1 > self.opts.num_test:
                    break
            else:
                length = len(label_data)
            progress_bar(index, length, "Loading {} dataset".format(mode))
            self._add_to_database(index, data_set, data_dir)

        equal_left, equal_right = cal_equal(6)
        print('\n%s Done %s' % ('=' * equal_left, '=' * equal_right))

    @staticmethod
    def _open_data_file(label_name, label_dir):
        """Open .csv file or .json file"""
        # judge label file suffixes
        suffixes = label_name.split('.')[-1]
        if suffixes.lower() == "csv":
            label_data = open_csv_file(label_dir)
        elif suffixes.lower() == "json":
            label_data = open_json_file(label_dir)
        else:
            raise IOError("[Error] Label file --> '{:s}' was not found...".format(label_dir))
        return label_data

    def __len__(self):
        """Get the batch length of dataset.
        Returns:
            length: the batch number of images."""
        return len(self.image_info)

    def __getitem__(self, index):
        """Return a data point and its metadata information. It
        usually contains the data itself and its metadata information.
        Inputs:
            index: a random integer for data indexing
        Returns:
            data: a tensor of image
            label: a tensor of label
        """
        # TODO(User) >>> return your own data information
        """Processing image"""
        assert os.path.isfile(self.image_info[index]["image_path"]), \
            self.image_info[index]["image_path"]+" was not found!"
        image = Image.open(self.image_info[index]["image_path"])
        image = self.transform(image)
        return image, self.label_info[index], self.image_info[index]["image_name"]
        # TODO(User): End


###############################################################
# Data Transforms Class
###############################################################
class DataTransforms(object):
    """This class includes common transforms for the dataset.
    Inputs:
        cfg: the total options

    Examples:
        <<< transform = DataTransforms(cfg)
            image = transform.get_transforms()(image)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = cfg.opts

    def __call__(self, image):
        """Apply all the transforms."""
        return self.get_transforms(image)

    @ staticmethod
    def prepare_image(image):
        """Process image for Image.open()."""
        # process the 4 channels .png
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))
        # process the 1 channel image
        elif image.mode != 'RGB':
            image = image.convert("RGB")
        return image

    def keep_ratio_resize(self, image, fill=(255, 255, 255)):
        # resize image with its h/w ratio, and padding the boundary
        old_size = image.size  # (width, height)
        ratio = min(float(self.opts.input_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        image = image.resize((new_size[1], new_size[0]), resample=Image.ANTIALIAS)  # w*h
        pad_w = self.opts.input_size[1] - new_size[1]
        pad_h = self.opts.input_size[0] - new_size[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        resize_image = Image.new(mode='RGB', size=(self.opts.input_size[1], self.opts.input_size[0]), color=fill)
        resize_image.paste(image, (left, top, new_size[1] + right, new_size[0] + bottom))  # w*h
        return resize_image

    def get_transforms(self, image):
        """Get the image transforms
        Returns:
            composes several transforms together
        """
        image = self.prepare_image(image)
        image = self.keep_ratio_resize(image)
        image = self.customer_transforms(image)
        # Convert to numpy and Normalize
        image = np.asarray(image) / 255.0
        return image

    # TODO(User) >>> create your own transforms customized functions
    def customer_transforms(self, image, fill=(255, 255, 255)):
        """Get customer image transforms
        Returns:
            a list several transforms together
        """
        prob = random.random()
        if self.opts.flip == 'vertical' and prob < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.opts.flip == 'horizontal' and prob < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.opts.rotate is not None:
            rotate = list(map(int, self.opts.rotate.split(',')))
            angle = random.uniform(rotate[0], rotate[1] + 1)
            image = image.rotate(angle, fillcolor=fill)

        return image
    # TODO(User): End


###############################################################
# DataLoader Class
###############################################################
class DataLoader(object):
    """Batch Data Generator for training or testing.
    Inputs:
        database: a object of BaseDataset
        cfg: the total options

    Examples:
        <<< train_loader = DataLoader(train_db, cfg)
            for data in train_loader.flow()
    """

    def __init__(self, database, cfg):
        self.database = database
        self.cfg = cfg
        self.opts = cfg.opts

    def __len__(self):
        """Get the batch length step of dataset.
        Returns:
            length: the number of steps."""
        return len(self.database) // self.opts.batch

    def flow(self):
        """Batch Data Generator.
        Yield:
            inputs: a batch of images with shape (b, h, w, c)
            outputs: a batch of labels with shape (b, num_classes)
        """
        b = 0  # batch item index
        step = 0
        image_index = -1
        image_ids = np.copy(self.database.image_ids)
        error_count = 0
        if self.cfg.mode == 'Test':
            batch = self.opts.num_test if self.opts.num_test < self.opts.batch else self.opts.batch
        else:
            batch = self.opts.batch

        # Keras requires a generator to run indefinitely.
        while True:
            try:
                # Init batch arrays
                batch_images = []
                batch_labels = []
                batch_names = []

                # TODO(User) >>> modify the inputs and outputs shape from BaseDataset
                # Batch full?
                while b < batch:
                    # Increment index to pick next image. Shuffle if at the start of an epoch.
                    image_index = (image_index + 1) % len(image_ids)
                    image_id = image_ids[image_index]
                    image_orig, label, name = self.database[image_id]
                    batch_images.append(image_orig)
                    batch_labels.append(label)
                    batch_names.append(name)
                    b += 1

                batch_images = np.asarray(batch_images, dtype=float)
                batch_labels = to_categorical(batch_labels, len(self.cfg.class_name))

                # TODO(User): END
                yield batch_images, batch_labels, batch_names
                # start a new batch
                b = 0
                step += 1
                if step >= len(self):
                    break

            except (GeneratorExit, KeyboardInterrupt):
                raise
            except ():
                # Log it and skip the image
                print("Error processing image !!!")
                error_count += 1
                if error_count > 5:
                    raise

