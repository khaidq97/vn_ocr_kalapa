export PYTHONPATH=.
MODEL_PATH=trained_models/seg2seg_ocr.pth
MODE=half
python3 tests/run_quantization.py --log-dir=logs/quantization \
    --model-path=$MODEL_PATH \
    --mode=$MODE 