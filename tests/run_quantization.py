import torch
import datetime
from pathlib import Path
import argparse
from lib.ocr_model.vietocr_model import VIETOCR

def convert_to_half_precession(model_path, save_dir):
    predictor = VIETOCR(model_path=str(model_path), device='cpu')
    model = predictor.model.model.eval()
    model.half()
    weight = model.state_dict()
    torch.save(weight, str(Path(save_dir)/'model_half.pth'))
    print('save model to {}'.format(str(Path(save_dir)/'model_half.pth')))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model-path', type=str,
                        default='path to model')
    args.add_argument('--log-dir', type=str,
                        default='logs/quantization')
    args.add_argument('--mode', type=str,
                        default='half')
    cfg = args.parse_args()
    
    log_dir = Path(cfg.log_dir)
    model_path = Path(cfg.model_path)
    mode = str(cfg.mode)
    
    # Log dict
    debug_dir = Path(log_dir)/ datetime.datetime.now().strftime('%Y%m%d_%H%M')
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == 'half':
        print('Convert to half precession mode')
        convert_to_half_precession(
            model_path=model_path,
            save_dir=debug_dir,
        )
    