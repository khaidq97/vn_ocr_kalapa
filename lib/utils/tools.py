
from pathlib import Path 
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

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

def insert_text_to_background(text_img, bg_img, x, y):
    """
    Insert text image to background image
    """
    text_img = text_img.copy()
    bg_img = bg_img.copy()
    # text_img[:,:,:3][text_img[:,:,3]==255] = [255,255,255]
    h, w = text_img.shape[:2]
    new_h, new_w = bg_img[:h, x:w+x].shape[:2]
    text_img = cv2.resize(text_img, (new_w, new_h))
    # alpha_normalized = text_img[:, :, 3] / 255.0
    
    # img_temp = text_img[:, :, :3] * alpha_normalized[:, :, None]
    # img_temp2 = bg_img[:h, x:w+x] * (1 - alpha_normalized[:, :, None])
    # h_,w_ = img_temp.shape[:2]
    # img_temp = cv2.resize(img_temp, (w_, h_))
    
    # bg_img[:h, x:w+x] = img_temp2 + img_temp
    bg_img[:h, x:w+x][text_img[:,:,3]==255] = 0
    return bg_img

def put_utf8_text(img, text, 
                  x=10, y=60, 
                  font_size=40, 
                  stroke_width=1,
                  color=(255, 0, 0), 
                  font_path='assets/couri.ttf'):
    """
    Put utf-8 text to image
    """
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font, fill=color, stroke_width=stroke_width)
    return np.array(img_pil)