import pyvips
from pathlib import Path
import sys

if __name__ == "__main__":

    path_src_str = sys.argv[1]
    path_src = Path(path_src_str)

    if len(sys.argv) > 2:
        file_name_dst = sys.argv[2]
    else:
        file_name_dst = path_src.stem + '.tiff'

    path_dst = path_src.parent / file_name_dst

    image = pyvips.Image.new_from_file(str(path_src))
    image.tiffsave(str(path_dst), pyramid=True, depth='onepixel', bigtiff=True)