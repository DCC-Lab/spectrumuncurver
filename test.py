from raytracing import *
path = ImagingPath()
path.label="Simple example"
path.append(Space(d=50))
path.append(Lens(f=50, diameter=25, label="First lens"))
path.append(Space(d=100))
path.append(Lens(f=100, diameter=25, label="Second lens"))
path.append(Space(d=50))
path.append(Aperture(12,"Spectrometer\nSlit"))
path.append(Space(d=50))
path.display()