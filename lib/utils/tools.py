
from pathlib import Path 

SUFFIX = ('.jpg', '.png', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')

def load_imgs_path(path):
    imgs_path = [x for x in Path(path).rglob('*') if x.suffix in SUFFIX]
    return imgs_path