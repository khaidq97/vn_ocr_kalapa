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

def run(controller, data_path, save_dir, logger):
    df_submitssion = pd.DataFrame(columns=['id', 'answer_ocr', 'answer_post', 'time_ocr', 'time_post'])
    debug_dir = save_dir / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    imgs_path = load_imgs_path(data_path)
    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(str(img_path))
        debug_img = img.copy()
        t0 = time.time()
        answer_ocr = controller.ocr_engine.run(img)[0]
        t1 = time.time()
        answer_post = controller.post_corrector(answer_ocr)
        t2 = time.time()
        time_ocr = t1 - t0 
        time_post = t2 - t1
        canvas_ocr = 255*np.ones_like(debug_img)
        canvas_post = 255*np.ones_like(debug_img)
        canvas_ocr = put_utf8_text(canvas_ocr, 'ocr :'+ answer_ocr, y=0)
        canvas_post = put_utf8_text(canvas_post, 'post:'+ answer_post, y=0)
        debug_img = np.concatenate([debug_img, canvas_ocr, canvas_post], axis=0)
        name = img_path.parent.name + '_' + img_path.stem
        cv2.imwrite(str(debug_dir/ (name + img_path.suffix)), debug_img)
        
    
        logger.info(f'{i+1}|{len(imgs_path)}|img:{img_path.name}|ans:{answer_ocr}=>{answer_post}|time_ocr:{time_ocr}s|time_post:{time_post}')
        id = img_path.parent.name + '/' + img_path.name
        df_submitssion.loc[i] = [id, answer_ocr, answer_post, time_ocr, time_post]
        
    df_submitssion.to_csv(Path(save_dir)/'logs.csv', index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log-dir', type=str, 
                      default='logs/post_correction')
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
    controller = Controller()
    
    run(
        controller=controller,
        data_path=data_path,
        save_dir=debug_dir,
        logger=logger
    )
    
    
    