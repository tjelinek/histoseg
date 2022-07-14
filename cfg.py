from pathlib import Path
import numpy as np

LOG_DIR = './logs/fit/'

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

mapping_classes_feit = {'adenocarcinoma': 'adenocarcinoma',
                        'artefact': 'unknown',
                        'blood': 'blood_and_vessels',
                        'blood_vessel': 'blood_and_vessels',
                        'connective_tissue': 'connective_tissue',
                        'fat': 'fat',
                        'inflammation_purulent': 'inflammation_purulent',
                        'ink': 'unknown',
                        'muscle_cross_section': 'muscle_cross_section',
                        'muscle_longitudinal_section': 'muscle_longitudinal_section',
                        'necrosis': 'necrosis',
                        'nerve': 'nerve',
                        'normal_mucosa': 'normal_mucosa',
                        'empty': 'empty',
                        'fragmucosa': 'unknown'}

selected_classes_feit = {'adenocarcinoma', 'blood_and_vessels', 'connective_tissue', 'fat', 'inflammation_purulent',
                         'muscle_cross_section', 'muscle_longitudinal_section', 'necrosis', 'nerve', 'normal_mucosa',
                         'empty',
                         # unknown
                         }

selected_classes_kather = {'01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE',
                           '08_EMPTY'}
