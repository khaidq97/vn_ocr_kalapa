import os 
from decouple import Config, RepositoryIni

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
INI_FILE = os.path.join(root_dir, 'settings.ini')
ini_config = Config(RepositoryIni(INI_FILE))

DEVICE = ini_config('DEVICE', cast=str, default='cpu')

OCR_MODEL_PATH = ini_config('OCR_MODEL_PATH', cast=str,
                           default=os.path.join(root_dir, 'trained_models', 'best_model.onnx'))

OCR_VOCAB_PATH = ini_config('OCR_VOCAB_PATH', cast=str,
                            default=os.path.join(root_dir, 'assets', 'vocab_short.txt'))

NGRAM_DATA_PATH = ini_config('NGRAM_DATA_PATH', cast=str,
                            default=os.path.join(root_dir, 'assets', 'postcorrection.csv'))

REPLACE_DATA_PATH = ini_config('REPLACE_DATA_PATH', cast=str,
                            default=os.path.join(root_dir, 'assets', 'replace_correction.csv'))

SENTENCE_DATA_PATH = ini_config('SENTENCE_DATA_PATH', cast=str,
                            default=os.path.join(root_dir, 'assets', 'sentence_correct.csv'))
