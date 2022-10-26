import os
from pathlib import Path

import pyvips

print(os.getcwd())
destination_folder = Path('/home/tomas/Projects/histoseg/data/Feit_colon-annotation')

os.chdir(destination_folder)

FACTOR = 2

tiff_paths = Path(os.getcwd()).rglob('*.tiff')

for tiff_path in tiff_paths:
    image_name = tiff_path.stem
    new_name = Path(str(image_name) + '_small.tiff')
    new_path = tiff_path.parent / new_name

    print(new_path)
    image = pyvips.Image.tiffload(str(tiff_path), page=FACTOR)
    image.tiffsave(str(new_path), bigtiff=True)
