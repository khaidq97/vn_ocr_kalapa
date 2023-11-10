from pathlib import Path 
from setuptools import setup
import pkg_resources

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent / 'winocr'  # root directory

def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        (List[str]): List of parsed requirements.
    """
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    
    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            if 'torch' in line:
                if 'torch' not in installed_packages:
                    requirements.append(line.split('#')[0].strip())  # ignore inline comments
    return requirements

setup(
    name="winocr",
    version="0.0.1",
    python_requires=">=3.6",
    packages=['winocr'] + [str(x) for x in Path('winocr').rglob('*/') if x.is_dir() and '__' not in str(x)],
    package_data={'winocr': ['assets/*']},
    include_package_data=True,
    install_requires=parse_requirements(PARENT / 'requirements.txt'),
    dependency_links=['--index-url https://download.pytorch.org/whl/cpu'],
)