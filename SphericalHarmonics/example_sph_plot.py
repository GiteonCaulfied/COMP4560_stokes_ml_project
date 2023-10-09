
import numpy
from scipy.special import sph_harm

import matplotlib.pyplot as P
import matplotlib.cm

if __name__ == '__main__':

    #
    # Load some data
    #
    out_sph_samples = numpy.load('../Data/Geoid/1k/output.npy')

    # Pick a set of coefficients
    out_sph = out_sph_samples[0, :]



    phi = numpy.linspace(0.0, numpy.pi, 100)
    theta = numpy.linspace(0.0, 2.0*numpy.pi, 100)
    theta, phi = numpy.meshgrid(theta, phi)

    # The Cartesian coordinates of the unit sphere
    x = numpy.sin(phi) * numpy.cos(theta)
    y = numpy.sin(phi) * numpy.sin(theta)
    z = numpy.cos(phi)

    accum = 0.0*sph_harm(0, 0, theta, phi)
    offset = 0
    for l in range(2, 8):
        for m in range(-l, l + 1):
            accum += out_sph[offset] * sph_harm(m, l, theta, phi)
            offset += 1

    use_3d = False
    if use_3d:
        fig = P.figure(figsize=P.figaspect(1.0))
        ax = fig.add_subplot(111, projection='3d')

        values = numpy.real(accum)
        vmin = numpy.min(values)
        vmax = numpy.max(values)

        nvalues = (values - vmin)/(vmax - vmin)
        norm = P.Normalize(vmin, vmax)
        
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=matplotlib.cm.seismic(nvalues))
        ax.set_axis_off()

    else:
        fig, ax = P.subplots()
        
        ax.imshow(numpy.real(accum), origin = 'lower', extent = [0, 360, -90, 90], cmap = 'seismic')

    P.show()
    
    
