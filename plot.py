import matplotlib.pyplot as plt
import nengo
from nengo.matplotlib import rasterplot
import numpy as np

spike_sorting = True
try:
    from Pycluster import kcluster
except:
    spike_sorting = False
    print "Couldn't import scipy.cluster. Can't sort spikes."

def remove_xlabels():
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticklabels([])

def nengo_plot_helper(offset, t, data, label='', removex=True, yticks=None, xlabel=None, spikes=False):
    ax = plt.subplot(offset)
    if spikes:
        rasterplot(t, data, label=label)
    else:
        plt.plot(t, data, label=label)

    plt.ylabel(label)
    plt.xlim(min(t), max(t))
    if yticks is not None:
        plt.yticks(yticks)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if removex:
        remove_xlabels()
    offset += 1

    return ax, offset

def nengo_stack_plot(offset, t, sim, probe, func=None, label='',
                     removex=False, yticks=None, slice=None, suppl=None, xlabel=None):
    if hasattr(probe, 'attr') and probe.attr == 'spikes':

        data, t = extract_probe_data(t, sim, probe, func, slice, suppl, spikes=True)
        return nengo_plot_helper(offset, t, data, label, removex, yticks, xlabel=xlabel, spikes=True)
    else:
        data, t = extract_probe_data(t, sim, probe, func, slice, suppl, spikes=False)
        return nengo_plot_helper(offset, t, data, label, removex, yticks, xlabel=xlabel, spikes=False)

def extract_probe_data(t, sim, probe, func=None, slice=None, suppl=None, spikes=False):
    if spikes:
        data = sim.data(probe)
        if slice is not None:
            data = data[slice, :]
            t = t[:data.shape[0]]
        return data, t

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

    newdata = apply_funcs(func, data, suppl)
    if newdata is not None:
        data = newdata

    if data.ndim > 1 and data.shape[1] > 600:
        mins = np.min(data, axis=1)[:, np.newaxis]
        maxs = np.max(data, axis=1)[:, np.newaxis]
        data = np.concatenate((mins, maxs), axis=1)

    return data, t

def apply_funcs(func, data, suppl=None):
    newdata = None

    if func is not None:
        if isinstance(func, list):
            func_list = func
        else:
            func_list = [func]

        for i, func in enumerate(func_list):
            if callable(func):
                if suppl:
                    fdata = np.array([func(d,s) for d,s in zip(data, suppl)])[:, np.newaxis]
                else:
                    fdata = np.array([func(d) for d in data])[:, np.newaxis]

                if newdata is None:
                    newdata = fdata
                else:
                    newdata = np.concatenate((newdata, fdata), axis=1)
    return newdata


def spike_sorter(spikes, k, slice=None, binsize=None):
    if not spike_sorting:
        print "Can't sort spikes."
        return t, spikes

    if slice is not None:
        cluster_data = spikes[slice]
    else:
        cluster_data = spikes


    if binsize is not None:
        new_data = np.zeros((0, cluster_data.shape[1]))

        for i in range(int(np.ceil(np.true_divide(cluster_data.shape[0], binsize)))):
            lo = i* binsize
            hi= min((i+1) * binsize, cluster_data.shape[0])
            new_data = np.concatenate((new_data, np.sum(cluster_data[lo:hi, :], axis=0)[np.newaxis,:]), axis=0)

        cluster_data = new_data

    labels, distortion, _ = kcluster(cluster_data.T, k, npass=30)
    print labels

    order = range(spikes.shape[1])
    order.sort(key = lambda l: labels[l])

    spikes = spikes[:, order]
    return spikes


