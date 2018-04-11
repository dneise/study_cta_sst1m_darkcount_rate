#!/usr/bin/env python
from tqdm import tqdm
from commandr import command, Run, SetOptions

import numpy as np
from scipy.ndimage.filters import convolve1d

from ctapipe.io import HDF5TableWriter
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.io.containers_calib import CalibrationHistogramContainer
from histogram.histogram import Histogram1D


@command('main')
def main(
    input_file,
    output_file,
    max_events=None,
    pixel_id=[...],
    integral_width=7,
    shift=0,
    pulse_finder_threshold=3.,
    use_digicam_baseline=False,
):

    baseline_span_ala_andrii = Histogram1D(
        data_shape=(1296,),
        bin_edges=np.arange(0, 50, 1)
    )

    random_charge = Histogram1D(
        data_shape=(1296,),
        bin_edges=np.arange(-50, 1000, 1)
    )

    if not use_digicam_baseline:
        events = calibration_event_stream(input_file,
                                          pixel_id=pixel_id,
                                          max_events=max_events)
        raw_histo = build_raw_data_histogram(events)
        save_container(raw_histo, output_file, 'histo', 'raw_lsb')

        events = calibration_event_stream(input_file,
                                          max_events=max_events,
                                          pixel_id=pixel_id)

        events = fill_histogram(events, 0, raw_histo)
        events = fill_electronic_baseline(events)
        events = subtract_baseline(events)
    else:
        events = calibration_event_stream(input_file,
                                          max_events=max_events,
                                          pixel_id=pixel_id)
        events = subtract_digicam_baseline(events)

    events = find_pulse_3(events, threshold=pulse_finder_threshold)
    events = compute_charge(events, integral_width, shift)
    events = fill_baseline_span_ala_andrii(events, baseline_span_ala_andrii)
    events = fill_random_charge(events, random_charge, 5)

    spe_charge = build_spe(events, bin_edges=np.arange(-20, 500, 1))

    save_container(spe_charge, output_file, 'histo', 'spe_charge')
    save_container(
        CalibrationHistogramContainer().from_histogram(baseline_span_ala_andrii),
        output_file,
        'histo',
        'baseline_span_ala_andrii'
    )
    save_container(
        CalibrationHistogramContainer().from_histogram(random_charge),
        output_file,
        'histo',
        'random_charge'
    )


def build_raw_data_histogram(events):

    for count, event in tqdm(enumerate(events)):

        if count == 0:

            n_pixels = len(event.pixel_id)
            adc_histo = Histogram1D(
                data_shape=(n_pixels, ),
                bin_edges=np.arange(0, 4095, 1),
                axis_name='[LSB]'
            )

        adc_histo.fill(event.data.adc_samples)

    return CalibrationHistogramContainer().from_histogram(adc_histo)


def fill_histogram(events, id, histogram):

    for event in events:

        event.histo[id] = histogram

        yield event


def fill_electronic_baseline(events):

    for event in events:

        event.data.baseline = event.histo[0].mode

        yield event


def subtract_baseline(events):

    for event in events:

        baseline = event.data.baseline

        event.data.adc_samples = event.data.adc_samples.astype(baseline.dtype)
        event.data.adc_samples -= baseline[..., np.newaxis]

        yield event


def subtract_digicam_baseline(events):
    for event in events:
        event.data.adc_samples = event.data.adc_samples.astype('f4')
        event.data.adc_samples -= event.data.digicam_baseline[:, np.newaxis]
        yield event


def find_pulse_3(events, threshold):
    w = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float32)
    w /= w.sum()

    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        pulse_mask = np.zeros(adc_samples.shape, dtype=np.bool)

        c = convolve1d(
            input=adc_samples,
            weights=w,
            axis=1,
            mode='constant',
        )
        pulse_mask[:, 6:-6] = (
            (c[:, :-2] <= c[:, 1:-1]) &
            (c[:, 1:-1] >= c[:, 2:]) &
            (c[:, 1:-1] > threshold)
        )[:, 5:-5]

        event.data.pulse_mask = pulse_mask

        yield event


def fill_baseline_span_ala_andrii(events, histo):

    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples
        content = adc_samples.max(axis=1) - adc_samples.min(axis=1)
        content = content[:, np.newaxis]
        histo.fill(content)

        yield event


def fill_random_charge(events, histo, integral_width):

    for count, event in enumerate(events):
        adc_samples = event.data.adc_samples

        possible_indices = np.arange(adc_samples.shape[1])
        random_places = np.random.np.random.choice(possible_indices[6:-6], 10)

        pulse_mask = np.zeros(adc_samples.shape, dtype=np.bool)
        pulse_mask[:, random_places] = True

        convolved_signal = convolve1d(
            adc_samples,
            np.ones(integral_width),
            axis=-1
        )

        charges = np.zeros(convolved_signal.shape) * np.nan
        charges[pulse_mask] = convolved_signal[pulse_mask]
        histo.fill(charges)

        yield event


def compute_charge(events, integral_width, shift):
    for count, event in enumerate(events):

        adc_samples = event.data.adc_samples
        pulse_mask = event.data.pulse_mask

        convolved_signal = convolve1d(
            adc_samples,
            np.ones(integral_width),
            axis=-1
        )

        charges = np.zeros(convolved_signal.shape) * np.nan
        charges[pulse_mask] = convolved_signal[
            np.roll(pulse_mask, shift, axis=1)
        ]
        event.data.reconstructed_charge = charges

        yield event


def build_spe(events, bin_edges=np.arange(-20, 500, 1)):

    for i, event in tqdm(enumerate(events)):

        if i == 0:

            n_pixels = len(event.pixel_id)

            spe_charge = Histogram1D(
                data_shape=(n_pixels,),
                bin_edges=bin_edges
            )

        spe_charge.fill(event.data.reconstructed_charge)

    spe_charge = CalibrationHistogramContainer().from_histogram(spe_charge)

    return spe_charge


def save_container(container, filename, group_name, table_name):

    with HDF5TableWriter(filename, mode='a', group_name=group_name) as h5:
        h5.write(table_name, container)


def entry():
    SetOptions(main='main')
    Run()

if __name__ == '__main__':
    entry()
