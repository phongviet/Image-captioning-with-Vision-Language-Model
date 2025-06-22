import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# 1. Đường dẫn tới file ground truth và file predictions
gt_file = 'data/COCO/annotations/captions_val2017.json'
pred_file = 'model_predictions.json'

# 2. Load ground truth
coco = COCO(gt_file)

# 3. Load predictions
with open(pred_file, 'r') as f:
    pred_data = json.load(f)

# Create COCO object from predictions
# Note: COCO requires predictions to have the specific format
coco_res = coco.loadRes(pred_data)

# 4. Tạo object COCOEvalCap with the COCO format predictions
coco_eval = COCOEvalCap(coco, coco_res)

# 5. Chỉ định image_ids nếu muốn (hoặc để mặc định là tất cả)
# coco_eval.params['image_id'] = coco.getImgIds()

# 6. Đánh giá
coco_eval.evaluate()

# 7. In kết quả
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score}")