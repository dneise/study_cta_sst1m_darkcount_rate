import h5py
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import trange
import numpy as np

from digicampipe.io.event_stream import calibration_event_stream


def single_gauss(x, x0, sigma, A):
    return (
        A / (np.sqrt(2 * np.pi) * sigma) *
        np.exp(
            -1/2 * ((x - x0) / sigma)**2
        )
    )


def fit_simple_gauss_to_maximum(x, y):
    p0 = (x[y.argmax()], 1, y.max())

    parameter_names = 'x0, sigma, A'.split(', ')

    popt, pcov = curve_fit(
        single_gauss,
        x,
        y,
        p0,
        sigma=np.sqrt(y).clip(1)
    )

    results = dict(zip(parameter_names, popt))

    results.update({
        n + '_error': v
        for (n, v) in zip(parameter_names, np.sqrt(pcov.diagonal()))
    })
    return results


def estimate_darkcount_rate_poisson(
    N_total,
    N_0,
    time_window
):
    return -np.log(N_0 / N_total) / time_window


def estimate_darkcount_rate_max_min(path, n_bins=50):
    f = h5py.File(path, 'r')
    bs = f['histo/baseline_span_ala_andrii'].value[0]
    x = (bs['bins'][1:] + bs['bins'][:-1]) / 2
    Y = bs['count']

    R = []
    for pixel_id in trange(bs['count'].shape[0]):
        try:
            results = fit_simple_gauss_to_maximum(x, Y[pixel_id])
            results['pixel_id'] = pixel_id
            results['dark_count_rate_MHz'] = estimate_darkcount_rate_poisson(
                N_total=Y[pixel_id].sum(),
                N_0 = results['A'],
                time_window = (n_bins * 4e-9) * 1e6  # 1e6 to convert in MHz
                )
            R.append(results)
        except (RuntimeError, ZeroDivisionError):
            pass

    return pd.DataFrame(R)


def estimate_darkcount_rate_random_charge(
    path,
    n_bins=5   # Horrible: magic number, comes from extract.py line: 59.
):

    f = h5py.File(path, 'r')
    bs = f['histo/random_charge'].value[0]
    x = (bs['bins'][1:] + bs['bins'][:-1]) / 2
    Y = bs['count']

    R = []
    for pixel_id in trange(bs['count'].shape[0]):
        try:
            results = fit_simple_gauss_to_maximum(x, Y[pixel_id])
            results['pixel_id'] = pixel_id
            results['dark_count_rate_MHz'] = estimate_darkcount_rate_poisson(
                N_total=Y[pixel_id].sum(),
                N_0=results['A'],
                time_window=(n_bins * 4e-9) * 1e6  # 1e6 to convert in MHz
                )
            R.append(results)
        except (RuntimeError, ZeroDivisionError):
            pass

    return pd.DataFrame(R)
