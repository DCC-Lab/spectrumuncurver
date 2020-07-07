from typing import List
from PIL import Image
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import asksaveasfilename


class SpectrumProcessor:

    def __init__(self, imagePath: str):
        self.spectrumImagePath = imagePath
        self.curbaturePeakZoneX = None
        self.curbaturePeakZoneY = None
        self.imPIL = None
        self.imMAT = None
        self.imArray = None
        self.imPlot = None
        self.method = 'maximum'

        self.gaussianPeakPos = []
        self.maximumPeakPos = []
        self.gaussianPixelDeviationX = []
        self.maximumPixelDeviationX = []

        self.shiftedImage = None
        self.shiftedPILImage = None
        self.load_image()

    def save_uncurved_image(self):
        path = asksaveasfilename()
        self.shiftedPILImage.save(path, compression='tif')

    def show_image_with_fit(self):
        self.uncurve_spectrum_image()
        plt.imshow(self.imArray, cmap='gray')
        print(len(np.linspace(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1], self.curbaturePeakZoneY[1]-self.curbaturePeakZoneY[0])))
        print(len(self.gaussianPeakPos))
        print(self.gaussianPeakPos)
        print(np.linspace(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1], self.curbaturePeakZoneY[1]-self.curbaturePeakZoneY[0]))

        plt.scatter([self.gaussianPeakPos], [np.linspace(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1], self.curbaturePeakZoneY[1]-self.curbaturePeakZoneY[0])], c='r', s=2, label="Gaussian fit")
        plt.scatter([self.maximumPeakPos], [np.linspace(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1],
                                                         self.curbaturePeakZoneY[1] - self.curbaturePeakZoneY[0])],
                    c='b', s=2, label="Gaussian fit")

        plt.show()
        # TODO make a superposition of points and image

    def show_uncurved_image(self):
        self.shiftedPILImage.show()

    def load_image(self):
        self.imPIL = Image.open(self.spectrumImagePath)
        self.imArray = np.array(self.imPIL)
        self.imMAT = plt.imread(self.spectrumImagePath)
        self.imPlot = plt.imshow(self.imMAT)

    def uncurve_spectrum_image(self, xlim: List, ylim: List, method='maximum'):
        self.curbaturePeakZoneX = xlim
        self.curbaturePeakZoneY = ylim
        self.method = method

        self.find_peak_position()
        self.find_peak_deviations()
        if self.method == 'maximum':
            result = self.correct_maximumDeviation_spectral_image()
        elif self.method == 'gaussian':
            result = self.correct_gaussianDeviation_spectral_image()
        return result

    def find_peak_position(self):
        for ypos in range(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1]):

            sectionyData = self.imArray[ypos][self.curbaturePeakZoneX[0]:self.curbaturePeakZoneX[1]]   # +1 because it doesn't include the given index
            sectionxData = np.linspace(self.curbaturePeakZoneX[0], self.curbaturePeakZoneX[1], len(sectionyData))
            pars, cov = curve_fit(f=self.gaussian, xdata=sectionxData, ydata=sectionyData,
                                  p0=[1, sectionxData[round(len(sectionxData)/2)], 1], bounds=(-np.inf, np.inf))
            stdevs = np.sqrt(np.diag(cov))
            maxIndex = sectionxData[np.argmax(sectionyData)]

            self.gaussianPeakPos.append(int(pars[1]))
            print(self.gaussianPeakPos)
            self.maximumPeakPos.append(int(maxIndex))

        print("GaussianPeakPos:", self.gaussianPeakPos)

    def find_peak_deviations(self):
        gaussianMidPosition = self.gaussianPeakPos[round(len(self.gaussianPeakPos) / 2)]
        maxMidPos = self.maximumPeakPos[round(len(self.maximumPeakPos) / 2)]
        print("gaussian peak avg position:", gaussianMidPosition)
        for ypos in range(self.curbaturePeakZoneY[1] - self.curbaturePeakZoneY[0]):
            xDev = self.gaussianPeakPos[ypos] - gaussianMidPosition
            self.gaussianPixelDeviationX.append(xDev)
            xDevMax = self.maximumPeakPos[ypos] - maxMidPos
            self.maximumPixelDeviationX.append(xDevMax)
            print(self.maximumPixelDeviationX)
            #print('Xdeviation:', xDev)

    def polyfit_peak_deviations(self):
        plt.plot(np.linspace(0,len(self.maximumPixelDeviationX), len(self.maximumPixelDeviationX)), self.maximumPixelDeviationX)

        x = np.linspace(0, len(self.maximumPixelDeviationX), len(self.maximumPixelDeviationX))
        y = self.maximumPixelDeviationX
        pars, cov = curve_fit(f=self.parabolic, xdata=x, ydata=y,
                              p0=[-1, 1, 1],
                              bounds=(-max(x), max(x)))
        stdevs = np.sqrt(np.diag(cov))

        devFity = self.parabolic(x, *pars)
        plt.plot(x, devFity, c='r')
        plt.show()

    def correct_maximumDeviation_spectral_image(self):
        self.shiftedImage = np.zeros(shape=(self.imPIL.height, self.imPIL.width))
        for ypos, i in enumerate(range(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1])):
            corr = -self.maximumPixelDeviationX[i]
            print(self.imArray[ypos][0:-corr])
            if corr >= 1:
                self.shiftedImage[ypos][corr:] = self.imArray[ypos][0:-corr]
            elif corr < 0:
                self.shiftedImage[ypos][0:corr-1] = self.imArray[ypos][-corr:-1]
            else:
                self.shiftedImage[ypos][::] = self.imArray[ypos][::]
        self.shiftedImage = self.shiftedImage/np.max(self.shiftedImage)
        self.shiftedPILImage = Image.fromarray(np.uint32(self.shiftedImage*63555))
        return self.shiftedImage

    def correct_gaussianDeviation_spectral_image(self):
        self.shiftedImage = np.zeros(shape=(self.imPIL.height, self.imPIL.width))
        for ypos, i in enumerate(range(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1])):
            corr = -self.gaussianPixelDeviationX[i]
            print(self.imArray[ypos][0:-corr])

            print("YPOS:", ypos)
            print("ARRAY", self.imArray[ypos])
            print("DEV", corr)
            if corr >= 1:
                self.shiftedImage[ypos][corr:] = self.imArray[ypos][0:-corr]
            elif corr < 0:
                self.shiftedImage[ypos][0:corr] = self.imArray[ypos][-corr:-1]
            else:
                self.shiftedImage[ypos][::] = self.imArray[ypos][::]

        return self.shiftedImage

    @staticmethod
    def gaussian(x, a, b, c):
        return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))

    @staticmethod
    def parabolic(x, a, b):
        return a*(x-b)**2


if __name__ == "__main__":
    PICorrector = SpectrumProcessor('data/glycerol_06_06_2020_2.tif')
    PICorrector.uncurve_spectrum_image([620, 670], [0, 398])


