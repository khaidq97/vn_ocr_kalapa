import time 
import pandas as pd
import datetime
from pathlib import Path 
import cv2 
import numpy as np
import argparse 
from lib.utils.logger import get_logger
from lib.controller import Controller
from lib.utils.metrics import calculate_score
from lib.utils.tools import load_imgs_path, put_utf8_text

def run(controller, data_file, imgs_path, logger, save_dir):
    save_dir = Path(save_dir)
    ocr_acc_true = save_dir / 'OCR' / 'True'
    ocr_acc_false = save_dir / 'OCR' / 'False'
    post_acc_true = save_dir / 'Post' / 'True'
    post_acc_false = save_dir / 'Post' / 'False'
    ocr_acc_true.mkdir(parents=True, exist_ok=True)
    ocr_acc_false.mkdir(parents=True, exist_ok=True)
    post_acc_true.mkdir(parents=True, exist_ok=True)
    post_acc_false.mkdir(parents=True, exist_ok=True)
    
    df_save = pd.DataFrame(columns=['id', 'label', 'ocr_pred', 'post_pred', 'ocr_score', 'post_score', 'ocr_acc', 'post_acc', 'time'])
    df = pd.read_csv(str(data_file))
    for i in range(len(df)):
        name = df.loc[i, 'image_name']
        label = df.loc[i, 'label']
        img_path = Path(imgs_path)/name
        img = cv2.imread(str(img_path))
        debug_img = img.copy()
        start_time = time.time()
        ocr_pred = controller.ocr_engine(img)
        post_pred = controller.post_corrector(ocr_pred)
        t = time.time() - start_time
        ocr_score = calculate_score(ocr_pred, label)
        post_score = calculate_score(post_pred, label)
        ocr_acc = 1 if ocr_score == 1 else 0
        post_acc = 1 if post_score == 1 else 0
        df_save.loc[i] = [name, label, ocr_pred, post_pred, ocr_score, post_score, ocr_acc, post_acc, t]
        logger.info(f'{i+1}|{len(df)}|{post_pred}|time:{t}')
        
        canvas = 255*np.ones_like(debug_img)
        canvas = put_utf8_text(canvas, post_pred, y=0)
        debug_img = np.concatenate([debug_img, canvas], axis=0)
        name = img_path.parent.name + '_' + img_path.stem
        
        if ocr_acc == 1:
            cv2.imwrite(str(ocr_acc_true/ (name + img_path.suffix)), debug_img)
        else:
            cv2.imwrite(str(ocr_acc_false/ (name + img_path.suffix)), debug_img)
        if post_acc == 1:
            cv2.imwrite(str(post_acc_true/ (name + img_path.suffix)), debug_img)
        else:
            cv2.imwrite(str(post_acc_false/ (name + img_path.suffix)), debug_img)
        
    logger.info(f'ocr_score: {df_save.ocr_score.mean()}')
    logger.info(f'ocr_acc: {df_save.ocr_acc.mean()}')
    logger.info(f'post_score: {df_save.post_score.mean()}')
    logger.info(f'post_acc: {df_save.post_acc.mean()}')
    logger.info(f'Average time: {df_save.time.mean()}')
    return df_save 




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log-dir', type=str, 
                      default='logs/benchmark')
    args.add_argument('--imgs-path', type=str,
                      default='Path to images')
    args.add_argument('--data-file', type=str,
                      default='data/test.csv')
    args.add_argument('--name', type=str,
                      default='')
    
    cfg = args.parse_args()
    
    log_dir = Path(cfg.log_dir)
    imgs_path = Path(cfg.imgs_path)
    data_file = Path(cfg.data_file)
    name = str(cfg.name)
    
    # Log dict
    debug_dir = Path(log_dir)/ (name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(name=__name__, 
                        mode='debug', 
                        save_dir=debug_dir)
    
    # Test
    controller = Controller()
    
    df = run(
        controller=controller,
        data_file=data_file,
        imgs_path=imgs_path,
        logger=logger,
        save_dir=debug_dir,
        )
    df.to_csv(Path(debug_dir)/'benchmark.csv', index=False)