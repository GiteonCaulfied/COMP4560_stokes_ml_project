
import numpy
import matplotlib.pyplot as P

if __name__ == '__main__':

    #
    # The data are stored in numpy binary files and are trivial to load into python
    #
    vinput = numpy.load('input.npy')
    voutput = numpy.load('output.npy')

    #
    # This initial dataset consists of 1000 randomly sampled models
    # The input vector is 257 and the output is 60, so these data arrays
    # are 1000 x 257 and 1000 x 60 respectively
    #
    print(vinput.shape)
    print(voutput.shape)

    # Just select one model to plot
    index = 123

    fig, ax = P.subplots(1, 2)

    ax[0].set_title('Input')
    ax[0].plot(vinput[index, :])

    ax[1].set_title('Output')
    ax[1].plot(voutput[index, :])

    P.show()
    

    
