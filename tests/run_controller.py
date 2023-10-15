import time 
import pandas as pd
import datetime
from pathlib import Path 
import cv2 
import argparse 
from lib.utils.logger import get_logger
from lib.controller import Controller
from lib.utils.tools import load_imgs_path

def run(controller, data_path, save_dir, logger):
    df_submitssion = pd.DataFrame(columns=['id', 'answer'])
    imgs_path = load_imgs_path(data_path)
    for i, img_path in enumerate(imgs_path):
        img = cv2.imread(str(img_path))
        start_time = time.time()
        answer = controller(img)
        end_time = time.time()
        logger.info(f'{i+1}|{len(imgs_path)}|img:{img_path.name}|ans:{answer}|Inference time: {end_time - start_time} s')
        id = img_path.parent.name + '/' + img_path.name
        df_submitssion.loc[i] = [id, answer]
        
    df_submitssion.to_csv(Path(save_dir)/'submission.csv', index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log-dir', type=str, 
                      default='logs/controller')
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
    
    
    