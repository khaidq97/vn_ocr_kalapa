import pandas as pd
import datetime
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def run(data_path, save_dir, train_ratio=0.9):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(columns=['name', 'text'])
    labels_file = list(Path(data_path).rglob('*.txt'))
    
    index = 0
    for file in labels_file:
        with open(str(file), 'r') as f:
            lines = f.readlines()
            name = file.name
            for line in lines:
                data = line.split('\t')
                name = data[0].strip()
                text = data[1].strip()
                df.loc[index] = [name, text]
                index += 1
    df.to_csv(str(save_dir/'data.csv'), index=False)
    train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=64)
    train_df.to_csv(str(save_dir/'train.csv'), index=False)
    test_df.to_csv(str(save_dir/'test.csv'), index=False)
    
    # convert to txt
    train_df.to_csv(str(save_dir/'train.txt'), index=False, sep='\t', header=False)
    test_df.to_csv(str(save_dir/'test.txt'), index=False, sep='\t', header=False)
            
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data-path', type=str,
                        default='path to model')
    args.add_argument('--log-dir', type=str,
                        default='logs/train_test')
    args.add_argument('--train-ratio', type=float,
                        default=0.9)
    cfg = args.parse_args()
    
    log_dir = Path(cfg.log_dir)
    data_path = Path(cfg.data_path)
    train_ratio = float(cfg.train_ratio)
    
    # Log dict
    debug_dir = Path(log_dir)/ datetime.datetime.now().strftime('%Y%m%d_%H%M')
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    run(
        data_path=data_path,
        save_dir=debug_dir,
        train_ratio=train_ratio
    )
    
    
    