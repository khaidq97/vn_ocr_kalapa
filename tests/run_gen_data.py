import pandas as pd
import datetime
from pathlib import Path 
import cv2 
import numpy as np
import argparse 
from lib.utils.logger import get_logger
from lib.utils.tools import load_imgs_path, SUFFIX, insert_text_to_background

def load_text_dict(text_path):
    text_dict = {}
    for text in text_path.glob('*'):
        text_dict[text.name] = [x for x in text.rglob('*') if x.suffix in SUFFIX]
    return text_dict


def run(background_path, text_trans_img_path, text_file, save_dir, logger, 
        num_imgs=1000, 
        bg_size=(1280,90), 
        text_size=(10,10),
        name='gen_data'):
    df = pd.read_csv(str(text_file))
    sentences = df['text'].values
    text_dict = load_text_dict(text_trans_img_path)
    background_list = load_imgs_path(background_path)
    
    save_img_dir = Path(save_dir)/name
    save_img_dir.mkdir(parents=True, exist_ok=True)
    
    label_file = Path(save_dir)/'label.txt'
    writer = open(str(label_file), 'w')
    for i in range(num_imgs):
        bg_file = np.random.choice(background_list)
        bg_img = cv2.imread(str(bg_file))
        bg_img = cv2.resize(bg_img, bg_size)
        sentence = np.random.choice(sentences)
        check = True
        for s in sentence.split():
            if s not in text_dict:
                check = False
                break
        if check:
            x,y = 10, 10
            try:
                for s in sentence.split():
                    text_trans_file = np.random.choice(text_dict[s])
                    text_img = cv2.imread(str(text_trans_file), cv2.IMREAD_UNCHANGED)
                    text_img = cv2.resize(text_img, text_size)
                    bg_img = insert_text_to_background(text_img, bg_img, x, y)
                    h,w = text_img.shape[:2]
                    x = x + w
                cv2.imwrite(str(save_img_dir/ f'gen_{i}.png'), bg_img)
                writer.write(f'{name}/gen_{i}.png\t{sentence}\n')
                logger.info(f'{i+1}|{num_imgs}|gen_{i}.png|{sentence}')
            except:
                print(f'Error: {i}')
            continue
    writer.close()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--log-dir', type=str, 
                      default='logs/gen_data')
    args.add_argument('--background-path', type=str,
                      default='path to data')
    args.add_argument('--text-file', type=str,
                      default='path to text file')
    args.add_argument('--text-trans-img-path', type=str,
                      default='path to text file')
    args.add_argument('--num-imgs', type=int,
                      default=1000)
    args.add_argument('--name', type=str,
                      default='gen_0')
    
    cfg = args.parse_args()
    
    log_dir = Path(cfg.log_dir)
    background_path = Path(cfg.background_path)
    text_file = str(cfg.text_file)
    text_trans_img_path = Path(cfg.text_trans_img_path)
    num_imgs = int(cfg.num_imgs)
    name = cfg.name
    
    # Log dict
    debug_dir = Path(log_dir)/ datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(name=__name__, 
                        mode='debug', 
                        save_dir=debug_dir)
    
    run(
        background_path=background_path,
        text_trans_img_path=text_trans_img_path,
        text_file=text_file,
        save_dir=debug_dir,
        logger=logger,
        num_imgs=num_imgs,
        bg_size=(1280,90),
        text_size=(90,70),
        name=name
    )
    
    
    