import time 
import pandas as pd
import datetime
from pathlib import Path 
import cv2 
import pandas as pd
import shutil
import argparse 
from lib.utils.logger import get_logger
from lib.text_detect import TextDetector
from lib.utils.tools import load_imgs_path, img_to_transparent, crop_img
from lib.utils.draw import draw

def crop_text_img_boxes(img, texts, boxes, save_dir, name):
    for text, box in zip(texts, boxes):
        crop_img_path = save_dir / text
        crop_img_path.mkdir(parents=True, exist_ok=True)
        croped_img = crop_img(img, box)
        cv2.imwrite(str(crop_img_path/ f"{name}_{text}_.png"), croped_img)

def gen_transparent_img(text_detector,
                        csv_file, 
                        imgs_path, 
                        save_dir, 
                        logger, 
                        input_type):
    df_save = pd.DataFrame(columns=['name', 'text', 'labels==boxes'])
    idx = 0
    df = pd.read_csv(str(csv_file))
    imgs_path = Path(imgs_path)
    detect_box_debug_dir = Path(save_dir)/'detect_box'
    detect_box_debug_dir.mkdir(parents=True, exist_ok=True)
    
    trans_img_dir = Path(save_dir)/'transparent'
    trans_img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(df)):
        try:
            name_ = df.loc[i, 'name']
            img_path = imgs_path / name_
            texts = df.loc[i, 'text'].strip().split()
            img = cv2.imread(str(img_path))
            boxes = text_detector(img)
            detect_box_img = draw(img, boxes=boxes)
            trans_img = img_to_transparent(img)
            
            if input_type == 'file':
                save_trans_path = save_dir 
                save_detect_box_debug_dir = detect_box_debug_dir / 'full_trans_img'
                name = img_path.stem
            else:
                save_trans_path = save_dir/ 'full_trans_img' / img_path.parent.name
                save_detect_box_debug_dir = detect_box_debug_dir/ img_path.parent.name
                # name = img_path.parent.name + '_' + img_path.stem
                name = '_'.join(name_.split('/'))[:-4]
            save_trans_path.mkdir(parents=True, exist_ok=True)
            save_detect_box_debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_trans_path/(img_path.stem + '.png')), trans_img)
            cv2.imwrite(str(save_detect_box_debug_dir/img_path.name), detect_box_img)
            
            if len(texts) == len(boxes):
                crop_text_img_boxes(
                    img=trans_img,
                    texts=texts,
                    boxes=boxes,
                    save_dir=trans_img_dir,
                    name=name
                )
                is_gen = 1
            else:
                is_gen = 0
            df_save.loc[idx] = [name_, ' '.join(texts), is_gen]
            idx += 1
            logger.info(f'{i+1}|{len(df)}|{is_gen}|{texts}|img:{img_path.name}|Time:{time.time()}')
        except:
            continue
    df_save.to_csv(str(save_dir / 'gen_word.csv'), index=False)
        
def split_char_image(cvs_file, transparent_img_path, save_dir, logger, input_type):
    df = pd.read_csv(str(cvs_file))
    for i in range(len(df)):
        img_path = transparent_img_path/ df.loc[i, 'name']
        img = cv2.imread(str(img_path))
        img = img_to_transparent(img)
        if input_type == 'file':
            save_path = save_dir 
        else:
            save_path = save_dir/ img_path.parent.name
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path/img_path.name), img)
        logger.info(f'{i+1}|{len(df)}|img:{img_path.name}') 

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log-dir', type=str, 
                      default='logs/gen_word')
    args.add_argument('--root-dir', type=str,
                      default='path to root dir')
    args.add_argument('--data-file', type=str,
                      default='path to data')
    args.add_argument('--name', type=str,
                      default='')
    args.add_argument('--input-type', type=str, 
                      default='folder', help='file or folder')
    
    cfg = args.parse_args()
    
    log_dir = Path(cfg.log_dir)
    root_dir = Path(cfg.root_dir)
    data_file = Path(cfg.data_file)
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
    
    gen_transparent_img(
        text_detector=text_detector,
        csv_file=data_file,
        imgs_path=root_dir,
        save_dir=debug_dir,
        logger=logger,
        input_type=input_type
    )
    
    
    