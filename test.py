# =========================test.py============================
# This module used to test the total project.
#
# Version: 1.0.0
# Date: 2019.08.07
# ============================================================
"""
=======================Test the model=========================
python test.py --model_name DemoModel --net_name DemoNet --dataroot multi_class_demo
               --batch 32 --test_label None --num_test 120
               --load_checkpoint best
==============================================================
"""
from option import TestOptions
from database import load_database
from model import BaseModel
from util import print_test_info, wrote_csv_file


def test_model():
    """Test the model"""
    # 1. Get Testing Options
    cfg = TestOptions()

    # 2. Load train and val Dataset
    test_loader = load_database(cfg)

    # 3. Create a Model
    model = BaseModel(cfg)

    # 4. Outer loop for one batch test sample
    loss = acc = 0.0
    for per_step, (images, labels, images_names) in enumerate(test_loader.flow()):
        results = []
        model.test(images=images, labels=labels, mode='test')
        loss += model.loss
        acc += model.metric

        if cfg.opts.test_label == 'None':
            predict = model.out.argmax(1)
            for i in range(len(predict)):
                results.append([images_names[i], predict[i].item()])
                wrote_csv_file(model.result_path, results, mode='a', show=False)
        print_test_info(cfg, [per_step, len(test_loader)], loss, acc)


if __name__ == "__main__":
    test_model()
