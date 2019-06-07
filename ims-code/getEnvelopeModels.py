'''Athanasios Anastasiou 13/12/2015
A simple function to get the "upper" and "lower" values envelope
from a time series'''

from numpy import array, sign, zeros
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,show,hold,grid, xlabel, ylabel, title, figure

def get_envelope_models(aTimeSeries, rejectCloserThan = 0):
    '''Fits models to the upper and lower envelope peaks and troughs.

    A peak is defined as a region where the slope transits from positive to negative (i.e. local maximum).
    A trough is defined as a region where the slope transits from negative to positive (i.e. local minimum).

    This example uses cubic splines as models.

    Parameters:

    aTimeSeries:      A 1 dimensional vector (a list-like).
    rejectCloserThan: An integer denoting the least distance between successive peaks / troughs. Or None to keep all.
    '''
    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0,]
    u_y = [aTimeSeries[0],]
    lastPeak = 0;

    l_x = [0,]
    l_y = [aTimeSeries[0],]
    lastTrough = 0;

    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.
    for k in range(1,len(aTimeSeries)-1):
        #Mark peaks
        if (sign(aTimeSeries[k]-aTimeSeries[k-1])==1) and (sign(aTimeSeries[k]-aTimeSeries[k+1])==1) and ((k-lastPeak)>rejectCloserThan):
            u_x.append(k)
            u_y.append(aTimeSeries[k])
            lastPeak = k;

        #Mark troughs
        if (sign(aTimeSeries[k]-aTimeSeries[k-1])==-1) and ((sign(aTimeSeries[k]-aTimeSeries[k+1]))==-1) and ((k-lastTrough)>rejectCloserThan):
            l_x.append(k)
            l_y.append(aTimeSeries[k])
            lastTrough = k

    #Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.
    u_x.append(len(aTimeSeries)-1)
    u_y.append(aTimeSeries[-1])

    l_x.append(len(aTimeSeries)-1)
    l_y.append(aTimeSeries[-1])

    #Fit suitable models to the data. Here cubic splines.
    u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)

    return (u_p,l_p)

if __name__ == "__main__":
    #A simple time series
    s = array([1,2,3,4,5,4,5,6,5,6,7,8,7,8,7,6,5,6,5,4,3,2,3,2,3,2,1])

    #Estimate models without rejecting any peak
    P = getEnvelopeModels(s)
    #Evaluate each model over the domain of (s)
    q_u = map(P[0],xrange(0,len(s)))
    q_l = map(P[1],xrange(0,len(s)))
    #Plot everything
    plot(s);xlabel('x');ylabel('y');title('Upper and lower envelopes including all peaks and troughs');hold(True);plot(q_u,'r');plot(q_l,'g');grid(True);hold(False);show();

    #Estimate models by rejecting peaks and troughs that occur in less than 5 samples distance
    P = getEnvelopeModels(s,5)
    #Evaluate each model over the domain of (s)
    q_u = map(P[0],xrange(0,len(s)))
    q_l = map(P[1],xrange(0,len(s)))

    #Plot everything
    figure();
    plot(s);xlabel('x');ylabel('y');title('Upper and lower envelopes including rejecting peaks and troughs occuring in less than 5 samples distance');hold(True);plot(q_u,'r');plot(q_l,'g');grid(True);show()
