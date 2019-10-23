# =====================util.visuals.py=========================
# This module contains useful visualization functions for directly
# Observe the operation of the model and the network structure.
#
# Version: 1.0.0
# Date: 2019.08.07
# =============================================================

import os
import time
from util import cal_equal, progress_bar, wrote_txt_file


###############################################################
# Print Procedure information
###############################################################
def print_train_info(val_flag, cfg, epoch, step, lr, loss=100.0, metric=0.0):
    """Print training epoch, step, loss, lr, and auc information on the screen.
    Inputs:
        val_flag: whether print the val info according to the epoch(bool)
        cfg: training options
        epoch: a list includes --> [start_epoch, per_epoch, total_epoch]
        step: a list includes --> [per_step, total_step]
        loss: a float includes --> train loss value
        lr: the current learning rate
        metric: the current batch training metric
    """
    message = ""
    # print '==== Start training ===='
    if epoch[1] == 1 and step[0] == 1:
        equal_left, equal_right = cal_equal(16)
        message += "\n" + "=" * equal_left + " Start Training " + "=" * equal_right
        # print '==== Start training ===='
    elif epoch[1] == epoch[0] and epoch[1] != 1 and step[0] == 1:
        equal_left, equal_right = cal_equal(19)
        message += "\n" + "=" * equal_left + " Continue Training " + "=" * equal_right
    # print '---- Epoch [1/10] ----'
    if (epoch[1] % cfg.opts.print_epoch == 0 or epoch[1] == epoch[0] or epoch[1] == epoch[2]) and step[0] - 1 == 0:
        val_flag = True
        info = " Epoch [{}/{}] ".format(epoch[1], epoch[2])
        equal_left, equal_right = cal_equal(len(info))
        message += "\n" + "-" * equal_left + info + "-" * equal_right
        message += "\n>>> Learning rate {:.7f}\n".format(lr)
    # print time, per_step, loss and acc
    if val_flag and (step[0] - 1 == 0 or step[0] == step[1] or step[0] % cfg.opts.print_step == 0):
        current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
        message += "{}  Step:[{}/{}]".format(current_time, step[0], step[1])
        message += " " * (len(str(step[1]))-len(str(step[0]))) + "  Loss:{:.4f}  ".format(loss)
        message += "ACC:{:.3f}%  ".format(metric*100)

    if len(message) != 0:
        print(message)
        if cfg.opts.save_train_log:
            mode = 'w' if epoch[1] == 1 and step[0] == 1 else 'a'
            wrote_txt_file(os.path.join(cfg.CHECKPOINT_DIR, 'Train_Log.txt'), message, mode=mode, show=False)
    return val_flag


def print_val_info(val_flag, cfg, step, loss=100.0, metric=0.0):
    """Print validate information on the screen.
    Inputs:
        val_flag: whether print the information or not(bool)
        cfg: training options
        step: a list includes --> [per_step, total_step]
        loss: a float includes --> val loss value
        metric: the current batch validating metric
    """
    info = message = ""
    if val_flag:
        if step[0] - 1 == 0:
            info += ">>> Validate on the val dataset ..."
            print(info)
            if cfg.opts.save_train_log:
                wrote_txt_file(os.path.join(cfg.CHECKPOINT_DIR, 'Train_Log.txt'), info, mode='a', show=False)
        progress_bar(step[0]-1, step[1], display=False)
        if step[0] >= step[1]:
            message += "\n>>> Loss:{:.4f}  ACC:{:.3f}%  ".format(loss/step[1], metric/step[1]*100)
            print(message)
            if cfg.opts.save_train_log:
                wrote_txt_file(os.path.join(cfg.CHECKPOINT_DIR, 'Train_Log.txt'), message, mode='a', show=False)


def print_test_info(cfg, step, loss=100.0, metric=0.0):
    """Print validate information on the screen.
    Inputs:
        cfg: training options
        step: a list includes --> [per_step, total_step]
        loss: a float includes --> val loss value
        metric: the current batch testing metric
    """
    info = message = ""
    if step[0] == 0:
        equal_left, equal_right = cal_equal(15)
        info += "\n" + "=" * equal_left + " Start Testing " + "=" * equal_right
        print(info)
        if cfg.opts.save_test_log:
            wrote_txt_file(os.path.join(cfg.CHECKPOINT_DIR, 'Test_log.txt'), info, mode='w', show=False)
    progress_bar(step[0], step[1], display=False)
    if cfg.opts.test_label != 'None':
        if step[0] + 1 >= step[1]:
            message += "\n>>> Loss:{:.4f}  ACC:{:.3f}%  ".format(loss / step[1], metric / step[1] * 100)
            print(message)
            if cfg.opts.save_test_log:
                wrote_txt_file(os.path.join(cfg.CHECKPOINT_DIR, 'Test_log.txt'), message, mode='a', show=False)

