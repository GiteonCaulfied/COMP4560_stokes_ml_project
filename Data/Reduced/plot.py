
import numpy
import matplotlib.pyplot as P

def mkpoints(v):

    xx = numpy.linspace(0, 1, v.size + 1)

    x = numpy.zeros((v.size * 2,))
    y = numpy.zeros((v.size * 2,))

    for i in range(v.size):
        x[2*i] = xx[i]
        x[2*i + 1] = xx[i + 1]

        y[2*i] = v[i]
        y[2*i + 1] = v[i]

    return x, y

def least_squares_G(i, o):

    GT, res, rank, sv = numpy.linalg.lstsq(i, o, rcond = None)

    return GT.T

if __name__ == '__main__':

    vinput = numpy.load('small_8_1k-inv.npy')
    voutput = numpy.load('small_8_1k-geoid.npy')

    N, _ = vinput.shape

    nmodels, _ = vinput.shape

    fig, ax = P.subplots(1, 2)

    ax[0].set_title('Input')
    ax[1].set_title('Output')

    # Just plot every 20th model input and output
    for i in range(1, nmodels, 20):

        ix, iy = mkpoints(vinput[i, :])
        
        ax[0].plot(ix, iy)

        ax[1].plot(voutput[i, :])

    G = least_squares_G(vinput, voutput)

    res = numpy.zeros((N,))
    for i in range(N):
        pred = G.dot(vinput[i, :])
        true = voutput[i, :]
        res[i] = numpy.linalg.norm(pred - true)

    fig, ax = P.subplots()
    ax.plot(numpy.sort(res))

    fig, ax = P.subplots(2, 2)

    testi = numpy.argmin(res)
    pred = G.dot(vinput[testi, :])
    true = voutput[testi, :]

    ix, iy = mkpoints(vinput[testi, :])
    ax[0][0].plot(ix, iy)

    ax[1][0].plot(pred)
    ax[1][0].plot(true)
    
    testi = numpy.argmax(res)
    pred = G.dot(vinput[testi, :])
    true = voutput[testi, :]

    ix, iy = mkpoints(vinput[testi, :])
    ax[0][1].plot(ix, iy)

    ax[1][1].plot(pred)
    ax[1][1].plot(true)


    P.show()
    

    
