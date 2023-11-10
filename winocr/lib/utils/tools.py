
from pathlib import Path 
import numpy as np
from PIL import ImageFont, ImageDraw, Image

SUFFIX = ('.jpg', '.png', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')

def load_imgs_path(path):
    imgs_path = [x for x in Path(path).rglob('*') if x.suffix in SUFFIX]
    return imgs_path

def crop_img(img, box):
    x1, y1, x2, y2 = box
    return img.copy()[int(y1):int(y2), int(x1):int(x2)]

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