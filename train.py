# ========================train.py============================
# This module used to train the total project.
#
# Version: 1.0.0
# Date: 2019.08.07
# ============================================================
"""
======================Train the model=========================
python train.py --model_name DemoModel --net_name DemoNet --dataroot multi_class_demo
                --batch 32 --epoch 15 --lr 1e-3 --gpu_ids 0
                --load_checkpoint scratch --flip horizontal
==============================================================
"""

from option import TrainOptions
from database import load_database
from model import BaseModel
from util import print_train_info, print_val_info


def val_model(cfg, model, val_loader, val_flag, per_epoch):
    """val the model"""
    save_metrics = {"LOSS": 0.0, "ACC": 0.0}
    # inner loop for one batch
    for per_step, (images, labels, _) in enumerate(val_loader.flow()):
        model.test(images=images, labels=labels, mode='val')
        save_metrics["LOSS"] += model.loss
        save_metrics["ACC"] += model.metric
        print_val_info(val_flag, cfg, [per_step + 1, len(val_loader)], save_metrics["LOSS"], save_metrics["ACC"])

    if "LOSS" in cfg.opts.save_metric:
        metrics = save_metrics["LOSS"] / len(val_loader)
        if metrics < model.BEST_METRIC:
            model.BEST_METRIC = metrics
            model.save_model(per_epoch, ["Bestval" + cfg.opts.save_metric, metrics])
    else:
        metrics = save_metrics["ACC"] / len(val_loader)
        if metrics > model.BEST_METRIC:
            model.BEST_METRIC = metrics
            model.save_model(per_epoch, ["Bestval" + cfg.opts.save_metric, metrics])


def train_model():
    """Train the model"""
    # 1. Get Training Options
    cfg = TrainOptions()

    # 2. Load train and val Dataset
    train_loader, val_loader = load_database(cfg)

    # 3. Create a Model
    model = BaseModel(cfg)

    # 4. Training
    for per_epoch in range(model.start_epoch+1, cfg.opts.epoch+1):
        val_flag = False
        save_metrics = {"LOSS": 0.0, "ACC": 0.0}
        # inner loop for one batch
        for per_step, (images, labels, _) in enumerate(train_loader.flow()):
            model.train(images=images, labels=labels)
            save_metrics["LOSS"] += model.loss
            save_metrics["ACC"] += model.metric
            val_flag = print_train_info(val_flag, cfg, [model.start_epoch+1, per_epoch, cfg.opts.epoch],
                                        [per_step + 1, len(train_loader)], model.lr, model.loss, model.metric)

        if cfg.opts.is_val:
            val_model(cfg, model, val_loader, val_flag, per_epoch)
        if per_epoch % cfg.opts.save_list == 0:
            model.save_model(per_epoch, ["train" + cfg.opts.save_metric,
                                         save_metrics[cfg.opts.save_metric]/len(train_loader)])
        model.update_lr(per_epoch+1)


if __name__ == "__main__":
    train_model()
