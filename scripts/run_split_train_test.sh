export PYTHONPATH=.
python3 tests/run_split_train_test.py \
    --data-path=/media/khaidq@kaopiz.local/hdd1/OCR/KALAPA_OCR_VN/data/training_data/annotations \
    --log-dir=logs/train_test \
    --train-ratio=0.9