import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve

def win_iter(l, wl=5):
    """
    Generate a sliding window of wl dimension
    
    Parameters
    ----------
    :param l: length of the array to split
    :param wl: window length
    
    Return
    ----------
    :return: at each iteration it returns the indices of the window
    """
    ss = l // wl
    splits = np.array_split(np.arange(l), ss)
    for s in splits:
        yield s


def window_local(x, wl=5):
    """
    Computes the local minima on a running window on the columns of a 2D array.
    
    Parameters
    ----------
    :param x: a 2D `numpy.array` containing the data
    :param wl: window length
    
    Return
    ----------
    :return: The index of the local minimum for each window.
    """
    n, p = x.shape
    ss = win_iter(p, wl)
    locmin = []

    for idx in ss:
        iwidx = np.argmin(x[:, idx], axis=1)
        locmin.append(idx[iwidx])

    locmin = np.asarray(locmin)
    return locmin.T


if __name__ == '__main__':

    # Set the working directory
    prjdir = '/home/v-gupta/wv2016/signal_processing/'

    # Read the data from a csv file. Columns separated by \t.
    # The first line of the file contains the scanned wavelengths
    tmpdata = np.loadtxt(os.path.join(prjdir, 'marzipan.csv'), delimiter='\t')
    wl = tmpdata[0]
    spectrum = tmpdata[1:]

    # Get dataset dimension
    n, p = spectrum.shape

    # Have a first look at our spectra
    plt.figure(1)
    for i in range(n):
        plt.plot(wl, spectrum[i, :], '--', label='sample %d' % i)
    plt.grid()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflection intensity')
    plt.show()

    # Find local minima
    idx = window_local(spectrum, wl=10)[0]

    # Determine the minimum points using first derivative
    diff_eq = np.diff(spectrum[0, :])
    sign_now = True
    minima_x = []
    minima_y = []
    for i in range(int(wl[0]), int(wl[0]) + len(diff_eq)):
        print (i)
        sign_prev = sign_now
        if diff_eq.item(i-wl[0]) > 0:
            sign_now = True
        else:
            sign_now = False
        if sign_prev and not sign_now:
            minima_x.append(i)
            minima_y.append(spectrum[0, i-wl[0]])
    #print (minima_x)
    #print (minima_y)
    print (wl)

    # Compute regression line for minima
    p2 = np.polyfit(minima_x, minima_y, deg=2)
    pf = np.poly1d(p2)
    print (p2)
    print (pf)

    # Plot and have a look at the data
    plt.figure(2)
    plt.plot(wl, spectrum[0, :], 'b', label='sample %d' % 0) #Line through local minima
    plt.plot(wl[idx], spectrum[0, idx], 'or', label='local minima') #Former local minima
    plt.plot(minima_x, minima_y, 'or', label = 'local minima') #New local minima
    plt.plot(wl, pf(wl), '--', label='Fitting') #Best fit line
    plt.plot(wl[1:], diff_eq, 'g', label = 'deriv of sample %d' % 0) #Graph of derivative
    plt.grid()
    plt.legend()
    plt.show()

