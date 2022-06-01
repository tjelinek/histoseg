import numpy as np

from staintools2.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools2.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools2.utils.optical_density_conversion import convert_OD_to_RGB
from staintools2.utils.get_concentrations import get_concentrations


class StainNormalizer(object):

    def __init__(self, method):
        self.stain_matrix_target_RGB = None
        self.maxC_target = None
        self.stain_matrix_target = None
        self.target_concentrations = None
        if method.lower() == 'macenko':
            self.extractor = MacenkoStainExtractor
        elif method.lower() == 'vahadane':
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception('Method not recognized.')

    def fit(self, target):
        """
        Fit to a target image.

        :param target: Image RGB uint8.
        :return:
        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        self.stain_matrix_target_RGB = convert_OD_to_RGB(self.stain_matrix_target)  # useful to visualize.

    def transform(self, I):
        """
        Transform an image.

        :param I: Image RGB uint8.
        :return: Transformed Image
        """
        stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (self.maxC_target / maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)
