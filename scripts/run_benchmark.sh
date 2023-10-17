export PYTHONPATH=.
python3 tests/run_benchmark.py \
    --name=training_val \
    --imgs-path=data/training_data/images \
    --data-file=data/test.csv