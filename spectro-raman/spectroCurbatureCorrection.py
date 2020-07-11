from typing import List
from PIL import Image
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import asksaveasfilename
from logging import getLogger
import argparse

log = getLogger(__name__)

__author__ = "Marc-André Vigneault"
__copyright__ = "Copyright 2020, Marc-André Vigneault", "DCCLAB", "CERVO"
__credits__ = ["Marc-André Vigneault"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Marc-André Vigneault"
__email__ = "marc-andre.vigneault.02@hotmail.com"
__status__ = "Production"

import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import sys


def init_logging(level):
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s\t\t (%(name)-15.15s) (thread:%(thread)d) (line:%(lineno)5d)\t\t[%(levelname)-5.5s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler in working directory
    logFolderPath = "."+os.sep+"log"
    if not os.path.exists(logFolderPath):
        os.makedirs(logFolderPath)
    handler = RotatingFileHandler(logFolderPath + os.sep + "{0}.log",
                                  maxBytes=10000, backupCount=5)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        "%(asctime)s\t\t (%(name)-25.25s) (thread:%(thread)d) (line:%(lineno)5d)\t\t[%(levelname)-5.5s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


init_logging(logging.INFO)
sys.excepthook = handle_exception


class SpectrumProcessor:

    def __init__(self, imagePath: str):
        self.spectrumImagePath = imagePath
        self.curbaturePeakZoneX = None
        self.pixelList = None
        self.curbaturePeakZoneY = None
        self.imPIL = None
        self.imMAT = None
        self.imArray = None
        self.imPlot = None
        self.method = 'maximum'

        self.gaussianPeakPos = []
        self.maximumPeakPos = []
        self.gaussianPixelDeviationX = []
        self.fittedGaussianPixelDeviationX = []
        self.maximumPixelDeviationX = []

        self.shiftedImage = None
        self.shiftedPILImage = None
        self.load_image()
        log.info("Class created successfully.")

    def save_uncurved_image(self):
        generalErrorMessage = "File was not saved. "
        try:
            self.shiftedPILImage = Image.fromarray(self.shiftedImage)
            path = asksaveasfilename()
            self.shiftedPILImage.save(path)
        except ValueError as e:
            if str(e) == "unknown file extension:":
                log.error("unknown file extension." + generalErrorMessage)
            else:
                log.error(generalErrorMessage + str(e))
        except AttributeError as e:
            if str(e) == "'NoneType' object has no attribute '__array_interface__'":
                log.error(generalErrorMessage + "No Image to save. Please proceed with loading and uncurving of the spectral data.")
            else:
                log.error(generalErrorMessage + str(e))

    def show_image_with_fit(self):
        plt.imshow(self.imArray, cmap='gray')
        plt.scatter([self.gaussianPeakPos], [np.linspace(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1], self.curbaturePeakZoneY[1]-self.curbaturePeakZoneY[0])], c='r', s=1, label="Gaussian fit")
        plt.scatter([self.maximumPeakPos], [np.linspace(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1],
                                                         self.curbaturePeakZoneY[1] - self.curbaturePeakZoneY[0])],
                    c='b', s=1, label="Gaussian fit")

        plt.show()

    def show_uncurved_image(self):
        self.shiftedPILImage = Image.fromarray(self.shiftedImage)
        self.shiftedPILImage.show()

    def show_curved_image(self):
        self.imPIL.show()

    def show_curabature(self):
        fig, ax = plt.subplot()
        ax.plot(self.gaussianPeakPos)

    def load_image(self):
        self.imPIL = Image.open(self.spectrumImagePath)
        self.imArray = np.array(self.imPIL)
        self.imMAT = plt.imread(self.spectrumImagePath)
        self.imPlot = plt.imshow(self.imMAT)

    def uncurve_spectrum_image(self, xlim: List, ylim: List, method='maximum'):
        self.curbaturePeakZoneX = xlim
        self.curbaturePeakZoneY = ylim
        self.pixelList = np.linspace(self.curbaturePeakZoneX[0], self.curbaturePeakZoneX[1], len(self.curbaturePeakZoneX)+1)
        self.method = method

        self.find_peak_position_on_each_row()
        self.find_peak_deviations_on_each_row()
        if self.method == 'maximum':
            result = self.correct_maximumDeviation_on_each_row()
        elif self.method == 'gaussian':
            result = self.correct_gaussianDeviation_on_each_row()
        return result

    def find_peak_position_on_each_row(self):
        for ypos in range(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1]):

            sectionyData = self.imArray[ypos][self.curbaturePeakZoneX[0]:self.curbaturePeakZoneX[1]+1]   # +1 because it doesn't include the given index
            sectionxData = np.linspace(self.curbaturePeakZoneX[0], self.curbaturePeakZoneX[1], len(sectionyData))
            log.debug("sectionxData:", sectionxData)
            pars, cov = curve_fit(f=self.gaussian, xdata=sectionxData, ydata=sectionyData,
                                  p0=[1, sectionxData[round(len(sectionxData)/2)], 1], bounds=(-np.inf, np.inf))
            stdevs = np.sqrt(np.diag(cov))
            maxIndex = sectionxData[np.argmax(sectionyData)]

            self.gaussianPeakPos.append(int(pars[1]))
            self.maximumPeakPos.append(int(maxIndex))

        log.info("GaussianPeakPos:{}".format(self.gaussianPeakPos))

    def find_peak_deviations_on_each_row(self):
        gaussianMidPosition = self.gaussianPeakPos[round(len(self.gaussianPeakPos) / 2)]
        maxMidPos = self.maximumPeakPos[round(len(self.maximumPeakPos) / 2)]
        log.info("gaussian peak avg position:{}".format(gaussianMidPosition))
        
        for ypos in range(self.curbaturePeakZoneY[1] - self.curbaturePeakZoneY[0]):
            xDevGaussian = self.gaussianPeakPos[ypos] - gaussianMidPosition
            self.gaussianPixelDeviationX.append(xDevGaussian)
            
            xDevMax = self.maximumPeakPos[ypos] - maxMidPos
            self.maximumPixelDeviationX.append(xDevMax)
        log.info("maximum peak average position:{}".format(self.maximumPixelDeviationX))

    def correct_maximumDeviation_on_each_row(self):
        self.shiftedImage = np.zeros(shape=(self.imPIL.height, self.imPIL.width))
        for ypos, i in enumerate(range(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1])):
            corr = -self.maximumPixelDeviationX[i]
            log.debug(self.imArray[ypos][0:-corr])
            if corr >= 1:
                self.shiftedImage[ypos][corr:] = self.imArray[ypos][0:-corr]
            elif corr < 0:
                self.shiftedImage[ypos][0:corr-1] = self.imArray[ypos][-corr:-1]
            else:
                self.shiftedImage[ypos][::] = self.imArray[ypos][::]
        self.shiftedImage = self.shiftedImage/np.max(self.shiftedImage)
        self.shiftedPILImage = Image.fromarray(np.uint32(self.shiftedImage*63555))
        return self.shiftedImage

    def correct_gaussianDeviation_on_each_row(self):
        self.shiftedImage = np.zeros(shape=(self.imPIL.height, self.imPIL.width))
        for ypos, i in enumerate(range(self.curbaturePeakZoneY[0], self.curbaturePeakZoneY[1])):
            corr = -self.gaussianPixelDeviationX[i]
            log.debug(self.imArray[ypos][0:-corr])

            log.debug("YPOS:", ypos)
            log.debug("ARRAY", self.imArray[ypos])
            log.debug("DEV", corr)
            if corr >= 1:
                self.shiftedImage[ypos][corr:] = self.imArray[ypos][0:-corr]
            elif corr < 0:
                self.shiftedImage[ypos][0:corr-1] = self.imArray[ypos][-corr:-1]
            else:
                self.shiftedImage[ypos][::] = self.imArray[ypos][::]

        return self.shiftedImage

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

    @staticmethod
    def gaussian(x, a, b, c):
        return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))

    @staticmethod
    def parabolic(x, a, b, c):
        return a*x**2 + b*x + c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to uncurve and graph spectrometer data.")
    parser.add_argument('imagePath', type=str,
                        help='The image that will be processed.')

    parser.add_argument('xyPeakLimits', type=int, nargs=4,
                        help='xmin, xmax, ymin, ymax position for peak curve analysis.')

    parser.add_argument('method', type=str,
                        help="chose between 'maximum', 'gaussian'")

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-U', '--uncurve', action='store_true',
                        help='Will prompt you to save the uncurved file.')

    group.add_argument('-S', '-superpose', action='store_true',
                        help='Will prompt you to save the curved image superposed with the peak data.')

    group.add_argument('-P', '-plot', action='store_true',
                       help='Will prompt you to save the uncurved and summed specturm plot.')

    args = parser.parse_args()

    print(args.imagePath)
    print(args.xyPeakLimits)
    try:

        spectrumUncurver = SpectrumProcessor(args.imagePath)
        spectrumUncurver.uncurve_spectrum_image([args.xyPeakLimits[0], args.xyPeakLimits[1]], [args.xyPeakLimits[2], args.xyPeakLimits[3]], args.method)
        spectrumUncurver.save_uncurved_image()

    except Exception as e:
        print(e)