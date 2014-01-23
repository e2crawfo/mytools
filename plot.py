import matplotlib.pyplot as plt
import nengo
from nengo.matplotlib import rasterplot
import numpy as np

def remove_xlabels():
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])

def spike_plot(offset, t, sim, probe, label='', removex=True, slice=None):

    data = sim.data(probe)

    if slice is not None:
        data = data[slice, :]
        t = t[:data.shape[0]]

    ax = plt.subplot(offset)
    rasterplot(t, data, label=label)
    plt.ylabel(label)
    if removex:
        remove_xlabels()
    offset += 1
    return ax, offset

def nengo_plot_helper(offset, t, sim, probe, label='', func=None, removex=False, slice=None):

    if isinstance(probe, list):
        data = []
        for i, p in enumerate(probe):
            if i == 0:
                data = sim.data(p)
            else:
                data = np.concatenate((data, sim.data(p)), axis=1)
    else:
        data = sim.data(probe)

    if data.ndim > 2:
        data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
    elif data.ndim == 1:
        data = data[:, np.newaxis]

    if slice is not None:
        data = data[slice, :]
        t = t[slice]

    if func is not None and callable(func):
        data = np.array([func(d) for d in data])

    if data.ndim > 1 and data.shape[1] > 600:
        mins = np.min(data, axis=1)[:, np.newaxis]
        maxs = np.max(data, axis=1)[:, np.newaxis]
        data = np.concatenate((mins, maxs), axis=1)

    ax = plt.subplot(offset)
    plt.plot(t, data, label=label)
    plt.ylabel(label)
    plt.xlim(min(t), max(t))
    if removex:
        remove_xlabels()
    offset += 1

    return ax, offset

def nengo_stack_plot(offset, t, sim, probe, func=None, label='', removex=False, slice=None):
    if hasattr(probe, 'attr') and probe.attr == 'spikes':
        return spike_plot(offset, t, sim, probe, label, removex, slice)
    else:
        return nengo_plot_helper(offset, t, sim, probe, label, func, removex, slice)

