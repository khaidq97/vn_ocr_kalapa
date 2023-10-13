import os 
from decouple import Config, RepositoryIni

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
INI_FILE = os.path.join(root_dir, 'settings.ini')
ini_config = Config(RepositoryIni(INI_FILE))