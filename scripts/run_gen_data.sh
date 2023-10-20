export PYTHONPATH=.
python3 tests/run_gen_data.py --log-dir=logs/gen_data \
    --background-path=/home/khai/Downloads/bacgrounds \
    --text-file=/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/assets/data.csv \
    --text-trans-img-path=/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/logs/gen_word/_20231019_2215/transparent \
    --num-imgs=10000 \
    --name=gen_0