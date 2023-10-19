import time 
import pandas as pd
import datetime
from pathlib import Path 
import cv2 
import shutil
import argparse 
from lib.utils.logger import get_logger
from lib.text_detect import TextDetector
from lib.utils.tools import load_imgs_path, save_yolo_file
from lib.utils.draw import draw

def run(text_detector, data_path, save_dir, logger, input_type):
    df = pd.DataFrame(columns=['id', 'coordinate'])
    yolo_dir = Path(save_dir)/'yolo'
    debug_dir = Path(save_dir)/'text_detect'
    yolo_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    imgs_path = load_imgs_path(data_path)
    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(str(img_path))
        start_time = time.time()
        boxes = text_detector(img)
        end_time = time.time()
        df.loc[i] = [img_path.name, str(boxes)]
        
        name = img_path.stem if input_type == 'file' else (img_path.parent.name + '_' + img_path.stem)
        save_yolo_file(
            file_name=yolo_dir/ 'labels' /(name + '.txt'),
            boxes=boxes,
            width=img.shape[1],
            height=img.shape[0]
        )
        img_yolo_dir = yolo_dir/ 'images'
        img_yolo_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, img_yolo_dir/ (name + img_path.suffix))
        # debug 
        debug_img = draw(img, boxes=boxes)
        cv2.imwrite(str(debug_dir/ (name + img_path.suffix)), debug_img)
        logger.info(f'{i+1}|{len(imgs_path)}|img:{img_path.name}|num box:{len(boxes)}|Inference time: {end_time - start_time} s')
    df.to_csv(Path(save_dir)/'data.csv', index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log-dir', type=str, 
                      default='logs/text_detect')
    args.add_argument('--data-path', type=str,
                      default='path to data')
    args.add_argument('--name', type=str,
                      default='')
    args.add_argument('--input-type', type=str, 
                      default='file', help='file or folder')
    
    cfg = args.parse_args()
    
    log_dir = Path(cfg.log_dir)
    data_path = Path(cfg.data_path)
    name = str(cfg.name)
    input_type = str(cfg.input_type)
    
    # Log dict
    debug_dir = Path(log_dir)/ (name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(name=__name__, 
                        mode='debug', 
                        save_dir=debug_dir)
    
    # Test
    text_detector = TextDetector()
    
    run(
        text_detector=text_detector,
        data_path=data_path,
        save_dir=debug_dir,
        logger=logger,
        input_type=input_type
    )
    
    
    