# spectrumuncuver
The `spectrumuncurver` python package is aimed at a specific problem which is often found
in spectrometers which is the curvature of the spectrum data caused by the lenses shape. Some spectrometers have 2D CCD
which allows the sum of the vertical pixels to achieve greater SNR. However, summing while the spectrum is curved
leads to a decrease of the spectral resolution. Thus, to achieve maximum resolution, one would want to uncuve the spectral data.

In addition to this main function, this package provides other function for graphing spectrum for articles.
You will find an extensive description of these function below, or more in details in the readthedocs.

Anyone who is willing to contribute is welcome to do so.

email: marc-andre.vigneault.02@hotmail.com



### installing the package

you can simply use `pip install spectrumuncurver` or for those without a `pip` link to `PATH`, `python -m pip install specturmuncurver`.



## using the package

To use the package, you have to follow 3 or 4 steps depending on your goal.
1. Create a ``su = SpectrumUncurver()`` object.
2. Load an image into the object. ``su.load_image(imagePath: str)``
3. Uncurve the image and choose the algorithm. 
``su.uncurve_spectrum_image(xlim:List, ylim:List, method='gaussian', fitted=False)``. 
You can choose from the following algorithms: ``maximum``, `gaussian`, `quadratic`. The ``fitted`` parameter will polyfit the found peaks to ensure
smooth uncurving, though not perfectly implemented.
4. Choose which function interests you from the following functions:

The package has (**or will have soon**) the following function:

### to save/show an image

- `save_uncurve_image()`
- `save_curved_image()`
- `save_curved_image_with_fit()`
- `save_curving_comparison_image()`
- `show_uncurve_image()`
- `show_curved_image()`
- `show_curved_image_with_fit()`
- `show_curving_comparison_image()`

### to save/show  a plot

- ``save_uncurved_spectrum_plot()``
- ``save_curved_spectrum_plot()``
- ``save_superposed_spectrum_plot()``
- ``save_curve_plot()``
- ``show_uncurved_spectrum_plot()``
- ``show_curved_spectrum_plot()``
- ``show_superposed_spectrum_plot()``
- ``show_curve_plot()``

Here is some results achieved with this package:

![image-20200712131837084](https://github.com/DCC-Lab/spectrumuncurver/raw/master/README.assets/firstResult.png)


