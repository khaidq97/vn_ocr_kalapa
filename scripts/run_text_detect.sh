export PYTHONPATH=.
# python3 tests/run_text_detect.py --input-type=file \
#     --data-path=/media/khaidq@kaopiz.local/hdd1/OCR/KALAPA_OCR_VN/v2/logs/gen_data/raw_2/images \
#     --log-dir=logs/text_detect 

python3 tests/run_text_detect.py --input-type=folder \
    --data-path=/media/khaidq@kaopiz.local/hdd1/OCR/KALAPA_OCR_VN/data/public_test/public_test/images \
    --log-dir=logs/text_detect 