# ===================model.base_model.py======================
# This module implements a base model for the project.
#
# Version: 1.0.0
# Date: 2019.08.07
# ============================================================

import os
import shutil
import time
import keras.callbacks as kc
import keras.layers as kl
import keras.optimizers as ko
import keras.backend as k
from keras.models import Model
from keras.utils import multi_gpu_model
from model import BaseNetwork
from util import cal_equal, list_sort
k.set_image_data_format('channels_last')  # (h, w, c)


###############################################################
# BaseModel Class
###############################################################
class BaseModel(object):
    """This class includes base processing for the model.
    Inputs:
        cfg: the total options

    Examples:
        <<< model = BaseModel(cfg)
            model.input(data)
            model.train()
            model.update_lr()
            model.save_model()
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = cfg.opts
        self.metric = 0.0
        self.loss = 1.0
        # Init training variable
        if cfg.mode == 'Train':
            self.best_cache = 'none'
            self.BEST_METRIC = cfg.BEST_METRIC
            self.lr = cfg.opts.lr
            self.start_epoch = cfg.opts.start_epoch - 1
            self.log_path = self._set_log_path()
            self.result_path = None
            self.out = None

        # Init gpu device
        self.gpu_ids = list(map(int, self.opts.gpu_ids.split(',')))

        # 1. Define network
        self.network = self._build_model() if len(self.gpu_ids) == 1 else \
            multi_gpu_model(self._build_model(), self.gpu_ids)
        # TODO(User): redefine the following >>> self.optimizer, self.criterion
        # 2. Define optimizer
        self.optimizer = ko.Adam(lr=self.opts.lr)
        # 3. Define loss
        self.criterion = "categorical_crossentropy"  # self.criterion = [focal_loss(alpha=.25, gamma=2)]
        # TODO(User): END

        self._display_network(verbose=self.opts.display_net)
        # 4. Load weights
        if cfg.opts.load_checkpoint != 'scratch':
            self.load_model()
        self._compile()

    def _build_model(self):
        """Build Network architecture."""
        # Inputs
        x_input = kl.Input(shape=self.opts.input_size, name="input_image")
        # Build the demo network
        if self.opts.net_name == "DemoNet":
            # TODO(User): select your network
            outputs = BaseNetwork(x_input=x_input, cfg=self.cfg)
            # TODO(User): END
        else:
            raise IOError("[Error] Please choose the right Network Architecture! --> '{:s}' was not found..."
                          .format(self.opts.net_name))
        model = Model(inputs=x_input, outputs=outputs[-1], name=self.opts.net_name)
        return model

    def _set_log_path(self):
        """Set TensorBoard output path."""
        log_path = os.path.join(self.cfg.CHECKPOINT_DIR, "log")
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.mkdir(log_path)
        return log_path

    def _compile(self):
        # Compile
        self.network.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['accuracy'])

    def train(self, images, labels=None):
        """Training flow of the whole model."""
        k.set_learning_phase(1)
        # TODO(User) >>> modify the Callbacks
        callbacks = [kc.TensorBoard(log_dir=self.log_path,
                                    histogram_freq=0,
                                    write_graph=True,
                                    write_images=False),
                     ]
        # TODO(User): End
        his = self.network.fit(images, labels, verbose=0)
        self.loss = his.history['loss'][0]
        self.metric = his.history['acc'][0]

    def test(self, images, labels=None, mode='val'):
        """val or test the whole model."""
        assert mode in ['val', 'test'], "mode in {}.test should be 'val or 'test'!".format(type(self).__name__)
        # evaluate
        if mode == 'val' or (mode == 'test' and self.opts.test_label != 'None'):
            his = self.network.evaluate(images, labels, verbose=0)
            self.loss = his[0]
            self.metric = his[1]
        else:
            self.out = self.network.predict(images, verbose=0)

    def update_lr(self, epoch):
        """Update learning rates for all the networks; called at the
        end of every epoch.
        Inputs:
            epoch: the current epoch
        """
        lr = self._create_scheduler(self.lr, epoch)
        k.set_value(self.network.optimizer.lr, lr)
        self.lr = k.get_value(self.network.optimizer.lr)

    def save_model(self, current_epoch, metrics=None):
        """Save all the networks to the disk.
        Inputs:
            current_epoch: current epoch of the total epoch
            current_step: current step of the current epoch
        """
        current_time = time.strftime("%m%d%H%M%S", time.localtime())
        save_filename = '[{}]_epoch:{}_{}:{:.3f}%_batch:{}_lr:{:f}_time:{}.h5' \
            .format(self.opts.net_name, current_epoch, metrics[0], metrics[1] * 100,
                    self.opts.batch, self.lr, current_time)
        if "Best" in metrics[0]:
            if os.path.isfile(os.path.join(self.cfg.CHECKPOINT_DIR, self.best_cache)):
                os.remove(os.path.join(self.cfg.CHECKPOINT_DIR, self.best_cache))
            save_filename = '[{}]_epoch:{}_{}:{:.3f}%_batch:{}_lr:{:f}_time:{}.h5'\
                .format(self.opts.net_name, current_epoch, metrics[0], metrics[1]*100,
                        self.opts.batch, self.lr, current_time)
            self.best_cache = save_filename
        save_path = os.path.join(self.cfg.CHECKPOINT_DIR, save_filename)
        self.network.save_weights(save_path)
        print(">>> Saving model -> %s" % save_filename)

    def load_model(self):
        """Load all the networks from the disk.
        Inputs:
            by_name: whether according its name
            exclude: list of layer names to exclude
        """
        file_path = self._find_checkpoint_path()
        self.network.load_weights(file_path, by_name=True, skip_mismatch=True)
        checkpoint_name = file_path.split(self.cfg.CHECKPOINT_DIR)[-1][1:-3]
        if checkpoint_name.startswith('[' + self.opts.net_name + ']'):
            self.BEST_METRIC = float(checkpoint_name.split('%')[0].split(':')[-1]) / 100
            self.start_epoch = int(checkpoint_name.split('epoch:')[-1].split('_')[0])
        print(">>> Loading model->%s" % (checkpoint_name + '.h5'))
        if self.cfg.mode == 'Train':
            if self.start_epoch >= self.opts.epoch:
                print("\n[Warning] Epoch:{} in options should be larger than Epoch:{} in checkpoints!"
                      .format(self.start_epoch, self.opts.epoch))
        elif self.cfg.mode == 'Test' and self.opts.test_label == 'None':
            self.result_path = os.path.join(self.cfg.CHECKPOINT_DIR, checkpoint_name + '.csv')
            open(self.result_path, 'w', newline='', encoding="utf-8-sig")

    def _find_checkpoint_path(self):
        if '.h5' not in self.opts.load_checkpoint:
            dir_names = []
            if 'best' in self.opts.load_checkpoint:
                for name in os.listdir(self.cfg.CHECKPOINT_DIR):
                    if name.startswith('[' + self.opts.net_name + ']') and 'Best' in name and name.endswith('.h5'):
                        dir_names.append(name)
                if len(dir_names) != 0:
                    save_path = os.path.join(self.cfg.CHECKPOINT_DIR, dir_names[0])
                    self.best_cache = dir_names[0]
                else:
                    raise IOError("[Error] No checkpoint file in {} ...".format(self.cfg.CHECKPOINT_DIR))
            else:
                for name in os.listdir(self.cfg.CHECKPOINT_DIR):
                    if name.startswith('[' + self.opts.net_name + ']') and name.endswith('.h5'):
                        dir_names.append(name)
                if len(dir_names) != 0:
                    dir_names = list_sort(dir_names, index=-1, mode="chars")
                    save_path = os.path.join(self.cfg.CHECKPOINT_DIR, dir_names[int(self.opts.load_checkpoint)])
                else:
                    raise IOError("[Error] No checkpoint file in {} ...".format(self.cfg.CHECKPOINT_DIR))
        else:
            save_path = os.path.join(self.cfg.CHECKPOINT_DIR, self.opts.load_checkpoint)
        return save_path

    def _create_scheduler(self, lr, epoch):
        """Create a learning rate scheduler."""
        if self.opts.lr_scheduler == 'power':
            lr = self.opts.lr * ((1 - float(epoch) / self.opts.epoch) ** self.opts.lr_power)

        elif self.opts.lr_scheduler == 'exp':
            lr = (float(self.opts.lr) ** float(self.opts.lr_power)) ** float(epoch + 1)

        elif self.opts.lr_scheduler == 'step':
            lr_step = list(map(int, self.opts.lr_step.split(',')))
            for step in lr_step:
                if epoch == step:
                    lr = lr / 10
        return lr

    def _display_network(self, verbose=False):
        """Print the total number of parameters and architecture
        in the network.
        Inputs:
            verbose: print the network architecture or not(bool)
        """
        equal_left, equal_right = cal_equal(22)
        print("\n" + "=" * equal_left + " Networks Initialized " + "=" * equal_right)
        if verbose:
            self.network.summary()
        if self.cfg.mode == 'Train':
            print('>>> [%s] Learning rate scheduler : %s ' % (self.opts.net_name, self.opts.lr_scheduler))
        print('>>> [%s] was created ...' % type(self).__name__)

