# ================database.__init__.py=======================
# This package includes some database modules: base dataset.

# Written by: GT
# Date: 2019.08.07
# ============================================================
__version__ = '1.0.0'

"""Database Modules"""
from .base_dataset import *


def load_database(cfg):
    """Create a dataset given the options.This is the main interface
    between this package and 'train.py'/'test.py'.
    Inputs:
        cfg: the total options
    Returns:
        train_loader: a train object of DataLoader
        val_loader: a val object of DataLoader
        test_loader: a test object of DataLoader
    """
    if cfg.mode == 'Train':
        train_db = BaseDataset(cfg, use_trans=True)
        train_db.load_data(mode='Train', shuffle=True)
        train_loader = DataLoader(database=train_db, cfg=cfg)

        if cfg.opts.is_val:
            val_db = BaseDataset(cfg)
            val_db.load_data(mode='Val', shuffle=True)
            val_loader = DataLoader(database=val_db, cfg=cfg)
        else:
            val_loader = None

        if '-1' in cfg.opts.save_epoch:
            cfg.opts.save_list = int(cfg.opts.save_epoch.split(',')[0].replace('-1', str(cfg.opts.epoch)))
        else:
            cfg.opts.save_list = int(cfg.opts.save_epoch.split(',')[0].replace(' ', ''))
        print(">>> [%s] was created ..." % type(train_db).__name__)
        return train_loader, val_loader

    elif cfg.mode == 'Test':
        test_db = BaseDataset(cfg)
        test_db.load_data(mode='Test', shuffle=False)
        test_loader = DataLoader(database=test_db, cfg=cfg)
        print(">>> [%s] was created ..." % type(test_db).__name__)
        return test_loader

    else:
        raise IOError("[Error] opts.mode --> '{:s}' in options should be 'Train' or 'Test'...".format(cfg.mode))




