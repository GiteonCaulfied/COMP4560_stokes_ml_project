
import numpy
from scipy.special import sph_harm

import pyshtools

import matplotlib.pyplot as P
import matplotlib.cm
import cartopy.crs as ccrs

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


    fig = P.figure()

    ax = fig.add_subplot(111, projection = ccrs.Robinson(central_longitude = 0.0))

    lons = theta * 180.0/numpy.pi
    lats = 90.0 - phi*180.0/numpy.pi
    
    img = ax.contourf(lons, lats, abs_image, 50, transform = ccrs.PlateCarree(), cmap = 'seismic')

    ax.coastlines()
    P.colorbar(img, ax=ax)
    
    P.show()
    
    
