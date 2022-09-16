import torch
import numpy as np
import cv2

from yolov6.layers.common import DetectBackend
from yolov6.data.datasets import LoadData
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression

def predict_paths(model_path, img_dir, img_size=640, conf_thres=0.5, iou_thres=0.45, 
                  classes=None, agnostic_nms=True, max_det=1000, half=False, device='cpu'):
    
    # Load model
    model = DetectBackend(model_path, device=device)
    model.model.float()

    # Load image files
    files = LoadData(img_dir)

    pred_lst = predict_images(files, model, img_size, conf_thres, iou_thres, 
                              classes, agnostic_nms, max_det, half, device)
    
    return pred_lst

def predict_images(files, model, img_size, conf_thres, iou_thres, classes, agnostic_nms, max_det, half, device):
    pred_lst = []
    for img_src, img_path, _ in files:
        img, img_src = precess_image(img_src, img_size, model.stride, half)
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]

        pred_results = model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]
        img_ori = img_src.copy()

        if len(det):
            det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round()

            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = None# f'{class_names[class_num]} {conf:.2f}'
                plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=generate_colors(class_num, True))

            # Find lowest bbox
            ymax_vals = [x[3] for x in np.asarray(det)]
            lowest_pos_idx = ymax_vals.index(max(ymax_vals))
            lowest_bbox = np.asarray(det[lowest_pos_idx])
            lowest_bbox_height = lowest_bbox[3] - lowest_bbox[1]

        pred_lst.append((img_ori, lowest_bbox_height, img_path))
    
    return pred_lst

def precess_image(img_src, img_size, stride, half):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src

def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape'''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
        
def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color