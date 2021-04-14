import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler

def get_hist(array, bins=10, xmin=None, xmax=None, underflow=False, overflow=False, mergeflowbin=True, normed=False,
            weights=None, **kwargs):
    r"""Plot histogram from input array.

    Arguments:
        array (np.ndarray): input array.
        bins (int, list or tuple of numbers, np.ndarray, bh.axis): bins
        weights (None, or np.ndarray): weights
        # normed (bool): deprecated.

    Returns:
        hist (boost_histogram.Histogram)
    """
    if isinstance(bins, int):
        if xmin is None:
            xmin = array.min()
        if xmax is None:
            xmax = array.max()
        width = 1.*(xmax-xmin)/bins
        if mergeflowbin and underflow:
            xmin += width
            bins -= 1
        if mergeflowbin and underflow:
            xmax -= width
            bins -= 1
        bins = bh.axis.Regular(bins, xmin, xmax, underflow=underflow, overflow=overflow)
    elif isinstance(bins, (list, tuple, np.ndarray)):
        if mergeflowbin and underflow:
            bins = bins[1:]
        if mergeflowbin and overflow:
            bins = bins[:-1]
        bins = bh.axis.Variable(bins, underflow=underflow, overflow=overflow)

    hist = bh.Histogram(bins, storage=bh.storage.Weight())
    if weights is None:
        weights = np.ones_like(array)
    hist.fill(array, weight=weights)
    return hist


def plot_hist(hists, normed=False, **kwargs):
    r"""Plot the histogram in the type of boost_histogram
    """
    
    if not isinstance(hists, (list, tuple)):
        hists = [hists]
    content = [h.view(flow=True).value for h in hists]
    bins = hists[0].axes[0].edges
    if 'bins' in kwargs:
        bins = kwargs.pop('bins')
    if 'yerr' in kwargs:
        yerr = kwargs.pop('yerr')
    else:
        yerr = [np.sqrt(h.view(flow=True).variance) for h in hists]
    if normed:
        for i in range(len(content)):
            contsum = sum(content[i])
            content[i] /= contsum
            yerr[i] /= contsum
    if len(hists) == 1:
        content, yerr = content[0], yerr[0]
    hep.histplot(content, bins=bins, yerr=yerr, **kwargs)

