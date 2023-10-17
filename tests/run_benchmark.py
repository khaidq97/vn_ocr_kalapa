import time 
import pandas as pd
import datetime
from pathlib import Path 
import cv2 
import argparse 
from lib.utils.logger import get_logger
from lib.controller import Controller
from lib.utils.metrices import calculate_score

def run(controller, data_file, imgs_path, logger):
    df_save = pd.DataFrame(columns=['id', 'label', 'predict', 'score', 'accuracy', 'time'])
    df = pd.read_csv(str(data_file))
    for i in range(len(df)):
        name = df.loc[i, 'name']
        label = df.loc[i, 'text']
        img_path = Path(imgs_path)/name
        img = cv2.imread(str(img_path))
        start_time = time.time()
        predict = controller(img)
        t = time.time() - start_time
        score = calculate_score(predict, label)
        accuracy = 1 if score == 1 else 0
        logger.info(f'{i+1}|{len(df)}|img:{name}|label:{label}|predict:{predict}|score:{score}|acc:{accuracy}|time:{t}')
        df_save.loc[i] = [name, label, predict, score, accuracy, t]
    logger.info(f'Accuracy: {df_save.accuracy.mean()}')
    logger.info(f'Average score: {df_save.score.mean()}')
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
        logger=logger
        )
    df.to_csv(Path(debug_dir)/'benchmark.csv', index=False)