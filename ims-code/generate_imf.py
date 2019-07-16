import numpy as np
import matplotlib.pyplot as plt
from getEnvelopeModels import get_envelope_models
from hht import *
#
name = "emd3"
path = "/Users/yapiachou/UiBergen/UiB-master/fig/{}.png".format(name)
t = np.linspace(-200,200,1000)
from pyhht.visualization import plot_imfs
import numpy as np
t = np.linspace(0, 1, 1000)
modes = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
x = modes + t
decomposer = EMD(x)
imfs = decomposer.decompose()
#plt.xlabel("x range")
plot_imfs(x, imfs, t)

#y = np.exp(-t/256.)*np.sin((np.pi*t/32.)+0.3*np.sin((np.pi*t/32.)))
#s = np.sin(2*np.pi*t/30.) + np.cos(2*np.pi*t/34.)
#P = get_envelope_models(x)
#plt.subplot(211)
#Evaluate each model over the domain of (s)
#q_u = list(map(P[0],range(0,len(x))))
#q_l = list(map(P[1],range(0,len(x))))
#m = list(map(lambda x1,x2: (x1+x2)/2., q_u,q_l))
#plot_imf(s)
#h1 = x-m

#plt.plot(t,x,label="signal S(t)")
#plt.plot(t,q_u,label="upper envelope u(t)")
#plt.plot(t,q_l,label="lower envelope l(t)")
#plt.plot(t,m, label="mean m(t)")
#plt.xlabel("x range")
#plt.ylabel("Amplitude")
#plt.xlabel("x range")
#plt.title("EMD process step 1")
#plt.title("a)", position=(0.9, 0.8))
#plt.legend(loc="upper center" ,bbox_to_anchor=(0.5, -0.05),ncol=4)
#plt.subplot(212)
#plt.plot(t,h1, label="s(t)-m(t)")
#plt.title("EMD process step 2")
#plt.title("b)", position=(0.9, 0.8))
#plt.legend(loc="best")

plt.savefig(path)
#plt.show()















#
