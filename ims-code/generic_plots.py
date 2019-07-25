import matplotlib.pyplot as plt
fig, ax = plt.subplots()
def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['right','top']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    #plt.xticks([]) # labels
    #plt.yticks([])
    #ax.xaxis.set_ticks_position('none') # tick markers
    #ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./40.*(ymax-ymin)
    hl = 1./40.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.1 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=hl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)


#ax.plot(x, y)
#ax.axhline(y=0.5, xmin=0.0, xmax=1.0, color='r')
def haar_phi(size,save=True):
    ax.hlines(y=1, xmin=0.0, xmax=1.0, color='b')
    ax.vlines(x=1, ymin=0.0, ymax=1.0, color='b',linestyles='dashed')
    #plt.axhline(y=1, xmin=0., xmax=0.9, linewidth=2)
    # RIGHT VERTICAL
    x=[1]; y = [1]
    #plt.axvline(x=0.402, ymin=0.4, ymax = 0.615, linewidth=2)
    plt.scatter(x, y, s=size, facecolors='none', edgecolors='b')
    plt.scatter([0], [1], s=size, edgecolors='b')
    plt.scatter([1], [0], s=size, edgecolors='b')
    plt.xlim([0,1.5])
    plt.ylim([0,1.5])
    plt.xlabel("t")
    #plt.title("The Haar scaling function " + r"$\phi$")
    plt.ylabel(r"$\phi(t)$")
    arrowed_spines(fig, ax)
    if save:
        plt.savefig("/Users/yapiachou/UiBergen/UiB-master/fig/haar-phi.png")
    else:
        plt.show()
    #plt.close()


def haar_psi(size,save=True):
    ax.hlines(y=1, xmin=0.0, xmax=0.5, color='b')
    ax.hlines(y=-1, xmin=0.5, xmax=1, color='b')
    ax.vlines(x=0.5, ymin=-1, ymax=1, color='b',linestyles='dashed')
    ax.vlines(x=1, ymin=-1, ymax=0, color='b',linestyles='dashed')

    plt.scatter([0.5], [1], s=size, facecolors='none', edgecolors='b')
    plt.scatter([1], [-1], s=size, facecolors='none', edgecolors='b')
    plt.scatter([0], [1], s=size, edgecolors='b',color="blue")

    plt.scatter([0.5],[-1], s=size, edgecolors='b',color="blue")
    plt.scatter([1],[0], s=size, edgecolors='b',color="blue")
    plt.xlim([0,1.5])
    plt.ylim([-1.5,1.5])
    plt.xlabel("t")
    #plt.title("The Haar wavelet function " + r"$\psi$")
    plt.ylabel(r"$\psi(t)$")
    #ax = plt.gca()
    #for side in ['right','top']:
        #ax.spines[side].set_visible(False)
    arrowed_spines(fig, ax)
    #ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),arrowprops=dict(arrowstyle="->"))
    if save:
        plt.savefig("/Users/yapiachou/UiBergen/UiB-master/fig/haar-psi.png")
    else:
        plt.show()
    #plt.close()


size = 30
#haar_phi(size,save=True)
haar_psi(size,save=True)
