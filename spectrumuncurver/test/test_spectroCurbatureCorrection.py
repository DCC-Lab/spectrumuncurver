from unittest import TestCase
from .. .spectrumuncurver import SpectrumUncurver
from PIL import Image
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt


class TestSpectrumProcessor(TestCase):
    def setUp(self) -> None:
        self.processor = SpectrumUncurver()

    def test_load_image(self):
        self.processor.load_image()
        loadedImage = Image.open('../data/glycerol_06_06_2020_2.tif')
        loadedArray = np.array(loadedImage)
        self.assertEqual(loadedArray, self.processor.imArray)


    def test_uncurve_spectrum_image(self):
        self.fail()

    def test_show_image_with_fit(self):
        self.fail()

    def test_find_peak_position(self):
        self.fail()

    def test_find_peak_deviations(self):
        self.fail()

    def test_correct_deviation_spectral_image(self):
        self.fail()

    def test_gaussian(self):
        result = self.processor.gaussian(0, 1, 2, 3)
        self.assertEqual(result, 0.8007374029168081, "Gaussian works")
