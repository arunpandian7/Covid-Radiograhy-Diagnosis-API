import cv2
import numpy as np
from mmcv import Config
from mmdet.apis import init_detector, inference_detector



config_path = "models/configs/cascade_rcnn_x101_32x4d_fpn_1x_coco.py"
model_checkpoint = "models/weights/cascade_rcnn_best_epoch.pth"

def load_model():
    print("Loading weights from:", model_checkpoint)
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, model_checkpoint, device='cuda:0')
    print("Model Loaded...")
    return model

def inference(img, threshold = 0.45):
    model = load_model()
    result = inference_detector(model, img)
    results_filtered = result[0][result[0][:, 4]>threshold]
    bboxes = results_filtered[:, :4]
    scores = results_filtered[:, 4]
    img = cv2.imread(img)
    for box in bboxes:
        img = draw_bbox(img, list(np.int_(box)), "Covid Detected",
                        (255, 243, 0))
    return img

def draw_bbox(image,
              box,
              label,
              color,
              label_size = 0.5,
              alpha_box = 0.3,
              alpha_label = 0.6):
    
    overlay_bbox = image.copy()
    overlay_label = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(),
                                              cv2.FONT_HERSHEY_SIMPLEX, label_size, 1)[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                  color, -1)
    cv2.addWeighted(overlay_bbox, alpha_box, output, 1-alpha_box, 0, output)
    
    cv2.rectangle(overlay_label, (box[0], box[1]-7-text_height),
                  (box[0]+text_width+2, box[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay_label, alpha_label, output, 1-alpha_label, 0, output)
    output = cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                           color, 2)
    cv2.putText(output, label.upper(), (box[0], box[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, label_size, (255, 255, 255), 1, cv2.LINE_AA)
    return output