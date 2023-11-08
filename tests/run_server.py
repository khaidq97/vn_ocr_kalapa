import time 
import datetime
from pathlib import Path 
import cv2 
import numpy as np
import io
from flask import Flask, request
from PIL import Image
import argparse 
from lib.utils.logger import get_logger
from lib.controller import Controller
from lib.utils.tools import put_utf8_text

args = argparse.ArgumentParser()
args.add_argument('--log-dir', type=str, 
                    default='logs/run_server')
args.add_argument('--url', type=str,
                    default='/ocr/run')
args.add_argument('--port', type=int,
                    default=5000)
args.add_argument('--mode', type=str, default='ocr',
                    choices=['ocr', 'controller'])
args.add_argument('--name', type=str,
                    default='')

cfg = args.parse_args()

log_dir = Path(cfg.log_dir)
url = cfg.url
port = int(cfg.port)
mode = cfg.mode
name = str(cfg.name)

# Log dict
debug_dir = Path(log_dir)/ (name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
debug_dir.mkdir(parents=True, exist_ok=True)

logger = get_logger(name=__name__, 
                    mode='debug', 
                    save_dir=debug_dir)

# Test
controller = Controller()
if mode=='ocr':
    engine = controller.ocr_engine
elif mode=='controller':
    engine = controller



app = Flask(__name__)

@app.route(url, methods=['POST'])
def run():
    if request.method != 'POST':
        return 
    if request.files.get('image'):
        im_file = request.files['image']
        im_name = request.form['name']

        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        debug_img = np.array(im)
  
        start_time = time.time()
        answer = controller(im)
        end_time = time.time()
        logger.info(f'img:{im_name}|ans:{answer}|Inference time: {end_time - start_time} s')
        
        canvas = 255*np.ones_like(debug_img)
        canvas = put_utf8_text(canvas, answer, y=0)
        debug_img = np.concatenate([debug_img, canvas], axis=0)
        save_img = debug_dir / 'debug'
        save_img.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_img/ im_name), debug_img)
        return answer

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=port, debug=True)
    
    
    