import time 
import pandas as pd
import datetime
from pathlib import Path 
import cv2 
import numpy as np
import argparse 
from lib.utils.logger import get_logger
from lib.controller import Controller
from lib.utils.tools import load_imgs_path, put_utf8_text

def run(ocr_model, data_path, save_dir, logger):
    df_submitssion = pd.DataFrame(columns=['id', 'answer'])
    debug_dir = save_dir / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    imgs_path = load_imgs_path(data_path)
    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(str(img_path))
        debug_img = img.copy()
        start_time = time.time()
        answer = ocr_model.run(img)[0]
        end_time = time.time()
        
        canvas = 255*np.ones_like(debug_img)
        canvas = put_utf8_text(canvas, answer, y=0)
        debug_img = np.concatenate([debug_img, canvas], axis=0)
        name = img_path.parent.name + '_' + img_path.stem
        cv2.imwrite(str(debug_dir/ (name + img_path.suffix)), debug_img)
        
    
        logger.info(f'{i+1}|{len(imgs_path)}|img:{img_path.name}|ans:{answer}|Inference time: {end_time - start_time} s')
        id = img_path.parent.name + '/' + img_path.name
        df_submitssion.loc[i] = [id, answer]
        
    df_submitssion.to_csv(Path(save_dir)/'submission.csv', index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log-dir', type=str, 
                      default='logs/ocr_model')
    args.add_argument('--data-path', type=str,
                      default='path to data')
    args.add_argument('--name', type=str,
                      default='')
    
    cfg = args.parse_args()
    
    log_dir = Path(cfg.log_dir)
    data_path = Path(cfg.data_path)
    name = str(cfg.name)
    
    # Log dict
    debug_dir = Path(log_dir)/ (name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(name=__name__, 
                        mode='debug', 
                        save_dir=debug_dir)
    
    # Test
    ocr_model = Controller().ocr_engine
    
    run(
        ocr_model=ocr_model,
        data_path=data_path,
        save_dir=debug_dir,
        logger=logger,
    )
    
    
    