from util.dataset_processing import fetch_files_prof_feit
from pathlib import Path
from util.data_manipulation_scripts import decompress_bz2_in_subfolder

destination_folder = Path('/home/tomas/Projects/histoseg/data/Feit_colon-annotation')
fetch_files_prof_feit('https://atlases.muni.cz/atlases/colonannot.html', destination_folder)


decompress_bz2_in_subfolder(destination_folder)