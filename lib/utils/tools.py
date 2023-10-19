
from pathlib import Path 
import cv2
import numpy as np

SUFFIX = ('.jpg', '.png', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')

def load_imgs_path(path):
    imgs_path = [x for x in Path(path).rglob('*') if x.suffix in SUFFIX]
    return imgs_path

def xyxy2yolo(box, width, height):
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2 / width
    y = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return (x, y, w, h)

def save_yolo_file(file_name, boxes, width, height, labels=None):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    with open(str(file_name), 'w') as f:
        for i in range(len(boxes)):
            box = xyxy2yolo(boxes[i], width, height)
            if labels is not None:
                label = labels[i]
            else:
                label = 0
            x, y, w, h = box
            f.write(f'{label} {x} {y} {w} {h}\n')
    
def sorted_bbox(boxes):
    """
    Sort boxes by x1
    """
    return sorted(boxes, key=lambda x: x[0]) if len(boxes) > 0 else boxes


def img_to_transparent(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    rgba_image = np.zeros((binary.shape[0], binary.shape[1], 4), dtype='uint8')
    rgba_image[:, :, 3] = binary
    rgba_image[:, :, :3] = img
    return rgba_image

def crop_img(img, box):
    x1, y1, x2, y2 = box
    return img.copy()[int(y1):int(y2), int(x1):int(x2)]