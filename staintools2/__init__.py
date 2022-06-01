import sys
if sys.version_info[0] < 3:
    raise Exception("Error: You are not running Python 3.")

from staintools2.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools2.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor

from staintools2.stain_normalizer import StainNormalizer
from staintools2.stain_augmentor import StainAugmentor
from staintools2.reinhard_color_normalizer import ReinhardColorNormalizer

from staintools2.preprocessing.luminosity_standardizer import LuminosityStandardizer
from staintools2.preprocessing.read_image import read_image
from staintools2.visualization.visualization import *
