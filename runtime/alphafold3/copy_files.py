import shutil
from importlib import resources
import alphafold3.constants.converters
import alphafold3
from pathlib import Path

def copy_files():
    # Define source file paths
    ccd_pickle_path = resources.files(alphafold3.constants.converters).joinpath('ccd.pickle')
    chemical_component_sets_pickle_path = resources.files(alphafold3.constants.converters).joinpath('chemical_component_sets.pickle')
    cpp_path = resources.files(alphafold3).joinpath('cpp.cpython-311-x86_64-linux-gnu.so')

    # Define target directories
    local_converters_dir = Path('./alphafold3/constants/converters/')
    local_main_dir = Path('./alphafold3/')

    # Ensure target directories exist
    local_converters_dir.mkdir(parents=True, exist_ok=True)
    local_main_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to target directories
    shutil.copy(ccd_pickle_path, local_converters_dir / 'ccd.pickle')
    shutil.copy(chemical_component_sets_pickle_path, local_converters_dir / 'chemical_component_sets.pickle')
    shutil.copy(cpp_path, local_main_dir / cpp_path.name)