
import numpy
from scipy.special import sph_harm

import pyshtools

import matplotlib.pyplot as P
import matplotlib.cm

def mk_image(coeff, phi, theta, latsamples = 100, lonsamples = 100):

    accum = 0.0*sph_harm(0, 0, theta, phi)
    offset = 0
    for l in range(2, 8):
        for m in range(-l, l + 1):
            accum += coeff[offset] * sph_harm(m, l, theta, phi)
            offset += 1

    return numpy.real(accum)

def mk_image_pyshtools(coeff, phi, theta, latsamples = 100, lonsamples = 100):

    sph = pyshtools.SHCoeffs.from_zeros(lmax = 7, normalization = 'ortho')

    offset = 0
    for l in range(2, 8):
        for m in range(-l, l + 1):
            sph.set_coeffs(coeff[offset], l, m)
            offset += 1

    image = sph.expand(lon = theta.flatten() * 180.0/numpy.pi,
                       lat = 90.0 - phi.flatten() * 180.0/numpy.pi)

    return image.reshape(theta.shape)

if __name__ == '__main__':

    #
    # Load some data
    #
    out_sph_samples = numpy.load('../Data/Geoid/1k/output.npy')

    # Pick a set of coefficients
    out_sph = out_sph_samples[0, :]

    error_sph = out_sph_samples[1, :]

    phi = numpy.linspace(0.0, numpy.pi, 100)
    theta = numpy.linspace(0.0, 2.0*numpy.pi, 100)
    theta, phi = numpy.meshgrid(theta, phi)

    # The Cartesian coordinates of the unit sphere
    x = numpy.sin(phi) * numpy.cos(theta)
    y = numpy.sin(phi) * numpy.sin(theta)
    z = numpy.cos(phi)

    abs_image = mk_image_pyshtools(out_sph, phi, theta)

    ref_image = mk_image_pyshtools(error_sph, phi, theta)
    err_image = abs_image - ref_image #mk_image_pyshtools((out_sph - error_sph), phi, theta)


    use_3d = False
    if use_3d:
        fig = P.figure(figsize=P.figaspect(1.0))
        ax = fig.add_subplot(111, projection='3d')

        vmin = numpy.min(abs_image)
        vmax = numpy.max(abs_image)

        nvalues = (abs_image - vmin)/(vmax - vmin)
        norm = P.Normalize(vmin, vmax)
        
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=matplotlib.cm.seismic(nvalues))
        ax.set_axis_off()

    else:
        fig, ax = P.subplots()
        
        ax.imshow(abs_image, origin = 'lower', extent = [0, 360, -90, 90], cmap = 'seismic')

        fig, ax = P.subplots()
        
        ax.imshow(numpy.abs(err_image), origin = 'lower', extent = [0, 360, -90, 90], cmap = 'magma')
        
    P.show()
    
    
