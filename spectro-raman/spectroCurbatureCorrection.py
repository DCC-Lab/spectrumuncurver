from typing import List
from PIL import Image
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt


class SpectrumProcessor:

    def __init__(self, imagePath: str, xlim: List, ylim: List):
        self.spectrumImagePath = imagePath
        self.correctionPeakZoneX = xlim
        self.correctionPeakZoneY = ylim
        self.imPIL = None
        self.imMAT = None
        self.imArray = None
        self.imPlot = None

        self.gaussianPeakPos = []
        self.maximumPeakPos = []
        self.gaussianPixelDeviationX = []
        self.maximumPixelDeviationX = []

        self.shiftedImage = None

    def load_image(self):
        self.imPIL = Image.open(self.spectrumImagePath)
        self.imMAT = plt.imread(self.spectrumImagePath)
        self.imArray = np.array(self.imPIL)
        self.imPlot = plt.imshow(self.imMAT)

    def uncurve_spectrum_image(self,):
        self.load_image()
        self.find_peak_position()
        self.find_peak_deviations()
        result = self.correct_deviation_spectral_image()

        return result

    def show_image_with_fit(self):
        fig, ax1 = plt.subplot()
        # plt.scatter(pars[1], ypos, c='r', s=10, label="Gaussian fit")
        # plt.scatter([maxIndex], [ypos], c='b', s=10, label="Maximum")
        # TODO make a superposition of points and image

    def find_peak_position(self):

        for ypos in range(self.correctionPeakZoneY[0], self.correctionPeakZoneY[1]):
            sectionyData = self.imArray[ypos][self.correctionPeakZoneX[0]:self.correctionPeakZoneX[1]]
            sectionxData = np.linspace(self.correctionPeakZoneX[0], self.correctionPeakZoneX[1], len(sectionyData))
            pars, cov = curve_fit(f=self.gaussian, xdata=sectionxData, ydata=sectionyData,
                                  p0=[1, sectionxData[round(len(sectionxData) / 2)], 1], bounds=(-np.inf, np.inf))
            stdevs = np.sqrt(np.diag(cov))
            maxIndex = sectionxData[np.argmax(sectionyData)]

            self.gaussianPeakPos.append([ypos, pars[1]])
            self.maximumPeakPos.append([ypos, maxIndex])

    def find_peak_deviations(self):
        gaussianMidPosition = self.gaussianPeakPos[round(len(self.gaussianPeakPos) / 2)][1]
        for ypos in range(self.correctionPeakZoneY[1] - self.correctionPeakZoneY[0]):
            xDev = self.gaussianPeakPos[ypos][1] - gaussianMidPosition
            self.gaussianPixelDeviationX.append(xDev)
            print('Xdeviation:', xDev)

    def correct_deviation_spectral_image(self):
        self.shiftedImage = np.zeros(shape=self.imPIL.shape)
        for ypos in range(self.correctionPeakZoneY[1] - self.correctionPeakZoneY[0]):
            dev = self.gaussianPixelDeviationX[ypos]
            self.shiftedImage[ypos][dev:] = self.imArray[0:-dev]

        return self.shiftedImage

    @staticmethod
    def gaussian(x, a, b, c):
        return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))


if __name__ == "__main__":
    PICorrector = SpectrumProcessor('data/glycerol_06_06_2020_2.tif', [620, 670], [0, 398])
    PICorrector.uncurve_spectrum_image()
    plt.imshow(PICorrector.shiftedImage)
