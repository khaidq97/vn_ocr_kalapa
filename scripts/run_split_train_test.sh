export PYTHONPATH=.
python3 tests/run_split_train_test.py \
    --data-path=/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/data/training_data/annotations \
    --log-dir=logs/train_test \
    --train-ratio=0.9