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

    def __init__(self, cfg, use_trans=False):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.opts = cfg.opts
        self.image_info = {}
        self.label_info = {}
        self.image_ids = []
        random.seed(cfg.opts.seed)
        # Instance a DataTransforms
        self.transform = DataTransforms(cfg, use_trans)

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

    def __init__(self, cfg, use_trans=False):
        self.cfg = cfg
        self.opts = cfg.opts
        self.use_trans = use_trans

    def __call__(self, image):
        """Apply all the transforms."""
        image = self.prepare_image(image)
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

    def get_transforms(self, image):
        """Get the image transforms.
        Returns:
            transformed
        """
        if self.use_trans:
            # customer image transforms
            if self.opts.flip == 'vertical':
                image = RandomVerticalFlip(p=0.5)(image)
            elif self.opts.flip == 'horizontal':
                image = RandomHorizontalFlip(p=0.5)(image)
            if self.opts.rotate is not None and random.random() <= 0.5:
                rotate = list(map(int, self.opts.rotate.split(',')))
                image = RandomRotation(degrees=rotate, expand=False)(image)
            if self.opts.translate is not None:
                image = RandomTranslate(self.opts.translate)(image)
            if self.opts.color_scale is not None:
                image = RandomColorChannel(self.opts.color_scale)(image)

        image = KeepRatioResize(resize_size=(self.opts.input_size[0], self.opts.input_size[1]))(image)
        # Convert to numpy and Normalize
        image = np.asarray(image) / 255.0
        return image


###############################################################
# Custom Data Transforms Class
###############################################################
# TODO(User) >>> create your own transforms customized class
class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Inputs:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Inputs:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Inputs:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Inputs:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
    
class RandomRotation(object):
    """Rotate the image by angle.
    Inputs:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
    """

    def __init__(self, degrees, resample=False, expand=False):
        if not isinstance(degrees, list):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand

    def __call__(self, img):
        """
        Inputs:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return img.rotate(angle, self.resample, self.expand)
    
    
class KeepRatioResize(object):
    """Keep the aspect ratio to resize given PIL Image without deforming the image.
    Inputs:
        resize_size(int or tuple): the output size of the resized image >>> (w, h)
        fill: (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively
    """

    def __init__(self, resize_size, fill=(0, 0, 0)):
        self.resize_size = resize_size if type(fill) != int else (resize_size, resize_size)
        self.fill = fill if type(fill) != int else (fill, fill, fill)
        assert len(self.resize_size) == 2, "resize_size must be int or tuple!"
        assert len(self.fill) == 3, "fill must be int or tuple!"

    def __call__(self, img):
        """
        Inputs:
            img (PIL Image): Image to be resize.
        Returns:
            img(PIL Image): Resized image.
        """
        # resize image with its h/w ratio, and padding the boundary
        old_size = img.size  # (width, height)
        ratio = min(float(self.resize_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        img = img.resize((new_size[0], new_size[1]), resample=Image.ANTIALIAS)  # w*h
        pad_h = self.resize_size[1] - new_size[1]
        pad_w = self.resize_size[0] - new_size[0]
        top = pad_h // 2
        left = pad_w // 2
        resize_image = Image.new(mode='RGB', size=(self.resize_size[1], self.resize_size[0]), color=self.fill)
        resize_image.paste(img, (left, top, left + img.size[0], top + img.size[1]))  # w*h
        return resize_image


class RandomTranslate(object):
    """Vertically or horizontally translate the given PIL Image randomly with a given coefficient.
    Inputs:
        translate(str or tuple): coefficient of the image being translate >>> (0.9, 1.2) or "0.9,1.2"
        p: the probability to translate the image
        fill: (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively
    """

    def __init__(self, translate, p=0.5, fill=(0, 0, 0)):
        self.p = p
        self.translate = translate if type(translate) != str else list(map(float, translate.split(',')))
        self.fill = fill if type(fill) != int else (fill, fill, fill)
        assert len(self.translate) == 2, "translate must be str or tuple with length:2"
        assert len(self.fill) == 3, "fill must be int or tuple!"

    def __call__(self, img):
        """
        Inputs:
            img (PIL Image): Image to be translate.
        Returns:
            img(PIL Image): Translated image.
        """
        if random.random() < self.p:
            return img
        ratio = random.uniform(self.translate[0], self.translate[1])
        position = random.randint(0, 1)  # 0 for vertical and 1 for horizontal
        (w, h) = img.size  # (width, height)
        left = 0 + abs(int(ratio * w - w)) * (1-position) if (ratio * w - w) < 0 else 0
        upper = 0 + abs(int(ratio * h - h)) * position if (ratio * h - h) < 0 else 0
        right = w - abs(int(ratio * w - w)) * (1-position) if (ratio * w - w) > 0 else w
        lower = h - abs(int(ratio * h - h)) * position if (ratio * h - h) > 0 else h
        img = img.crop(box=(left, upper, right, lower))
        resize_image = Image.new(mode='RGB', size=(h, w), color=self.fill)
        resize_image.paste(img, (left, upper, right, lower))  # w*h
        return resize_image


class RandomColorChannel(object):
    """Scaling the given PIL Image channels randomly with a given coefficient.
    Inputs:
        color_scale(str or tuple): coefficient of the image being scale >>> (0.6, 1.4) or "0.6,1.4"
        p: the probability to translate the image
    """

    def __init__(self, color_scale, p=0.5):
        self.p = p
        self.color_scale = color_scale if type(color_scale) != str else list(map(float, color_scale.split(',')))
        assert len(self.color_scale) == 2, "color_scale must be str or tuple with length:2"

    def __call__(self, img):
        """
        Inputs:
            img (PIL Image): Image to be scale.
        Returns:
            img(PIL Image): Scaled image.
        """
        # if random.random() < self.p:
        #     return img
        scale = np.ones(3)
        for i in range(3):
            scale[i] = random.uniform(self.color_scale[0], self.color_scale[1] + 0.1)
        img = np.asarray(img)
        # en = ImageEnhance.Color(img)
        # img = en.enhance(1.4)
        img = img * scale
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(img)
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

