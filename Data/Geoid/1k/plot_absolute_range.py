
import numpy
import matplotlib.pyplot as P

if __name__ == '__main__':

    vinput = numpy.load('input.npy')
    voutput = numpy.load('output.npy')

    nmodels, _ = vinput.shape

    fig, ax = P.subplots(1, 2)

    ax[0].set_title('Input')
    ax[1].set_title('Output')

    # Just plot every 20th model input and output
    for i in range(1, nmodels, 20):
    
        ax[0].plot(vinput[i, :])

        ax[1].plot(voutput[i, :])

    P.show()
    

    
