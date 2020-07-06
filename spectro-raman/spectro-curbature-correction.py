from PIL import Image
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt


im = plt.imread('glycerol_06_06_2020_2.tif')
imarray = np.array(im)
implot = plt.imshow(im)

def gaussian(x, a, b, c):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))

fig, ax1 = plt.subplots(1, 1)
print(imarray[200, :])

xmin = 620
xmax = 700
ymin = 0
ymax = 398
sectionyData = imarray[0][xmin:xmax]
sectionxData = np.linspace(xmin, xmax, len(sectionyData))

for ypos in range(ymin, ymax):
    sectionyData = imarray[ypos][xmin:xmax]
    sectionxData = np.linspace(xmin, xmax, len(sectionyData))
    pars, cov = curve_fit(f=gaussian, xdata=sectionxData, ydata=sectionyData, p0=[1, sectionxData[round(len(sectionxData)/2)], 1], bounds=(-np.inf, np.inf))
    stdevs = np.sqrt(np.diag(cov))
    maxIndex = sectionxData[np.argmax(sectionyData)]
    print(pars[1], sectionxData[np.argmax(sectionyData)])

    plt.scatter([pars[1]], [ypos], c='r', s=10, label="Gaussian fit")
    plt.scatter([maxIndex], [ypos], c='b', s=10, label="Maximum")
    plt.legend()

plt.show()

