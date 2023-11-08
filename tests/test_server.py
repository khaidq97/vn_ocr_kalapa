import pprint
import requests
from pathlib import Path

url = 'http://localhost:5000/ocr/run'
img_path = '/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/dataset/kalapa/public_test/public_test/images/1/0.jpg'

with open(img_path, 'rb') as f:
    img = f.read()
    
response = requests.post(url, files={'image': img}, 
                         data={'name': Path(img_path).name})
print(response.text)