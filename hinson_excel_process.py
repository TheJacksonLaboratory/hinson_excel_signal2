# Script for processing Hinson Lab excel spreadsheets with time series data
# Written by Dave Mellert in Research IT for The Jackson Laboratory
#
# Script takes an xlsx file with time series data, with a subset of columns
# laid out like 't1', 't2', 't3', etc. Columns must be ordered continuously
# left to right.
#
# Output is a copy of the original xlsx file with appended columns of
# summary data. Summaries are of the first peak.
#
# 
#
# Summary columns:
#     -peak_magnitude: magnitude of signal at peak
#     -area_under_peak: integral of peak between left and right base
#     -peak_duration: interval (n time samples) between left and right base
#     -rise: interval between left base and peak
#     -decay: interval between peak and right base
#     -half_max_up: interpolated timepoint when increasing signal is half peak
#                   value.
#     -peak_values: signal between left and right base, with padding
#     - half_max_down: interpolated timepoint when decreasing signal is half
#                      peak value.


import logging
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from scipy.signal import medfilt, find_peaks
from scipy.interpolate import interp1d
from os.path import basename

def smooth_signal(signal, window_size):
    """Return a smoothed signal
    Args
    ----
        signal: 1D numpy array
        window_size: size of sliding window for taking mean. Larger
                     values will produce more smoothing

    Returns
    -------
        smoothed: 1D numpy array that has been smoothed
    """
    # Window must be an odd numbered integer
    if window_size != int(window_size):
        raise ValueError("window size must be an integer")
    if window_size % 2 == 0:
        raise ValueError("window size must be odd")
    window_size = int(window_size)

    # Perform smoothing operation by convolving an array of ones
    kernel = np.ones(window_size)
    pad_width = int((window_size - 1) / 2)
    signal = np.pad(signal, pad_width=pad_width, mode='edge')
    smoothed = np.convolve(signal, kernel, mode='valid') / window_size
    return smoothed

def create_baseline(series):
    """return a constant value to serve as signal baseline
    Args
    ----
        series: 1D numpy array
    
    Returns
    -------
        baseline: floating point number
    """
    # Normalize the signal to range from 0 to 1
    # Note that this is a vectorized operation. "series" is the entire 1D array
    # whereas series.min() and series.max() are constants.
    series_norm = (series - series.min()) / (series.max() - series.min())

    # This is using scipy.signal.find_peaks to find peaks. Because all signals
    # are 0-to-1 normalized, the prominence parameter should work pretty well
    # for all signals.
    peaks = find_peaks(series_norm, prominence=0.2, wlen=35)

    # If if the first peak starts at 0, it is probably cut off. In that case,
    # take the second peak.
    if len(peaks[0]) == 0:
        logging.warning('BASELINE: No peaks found, setting baseline to 0')
        return np.float(0)

    if peaks[1]['left_bases'][0] == 0 and len(peaks[0]) > 1:
        peak_index = 1
    elif peaks[1]['left_bases'][0] == 0 and len(peaks[0]) == 1:
        peak_index = 0
        logging.warning('BASELINE: Signal only contains one peak,'
                        ' which starts at 0')
    else:
        peak_index = 0

    # The value of the un-normalized signal at the start of the first peak is
    # considered as baseline.
    left_base = peaks[1]['left_bases'][peak_index]
    baseline = series[left_base]
    return baseline

def deltaf(series, baseline):
    """calculate deltaf over f for a signal (1D array)"""
    deltaf = series - baseline
    return deltaf / baseline

def normalize_df(series):
    """convenience function to normalize a signal
    (i.e., rescale to range from 0 to 1)
    """
    series_norm = (series - series.min()) / (series.max() - series.min())
    return series_norm

def first_good_peak(peaks):
    """return only the relevant peak information in the form of a dict."""
    
    if len(peaks[0]) == 0:
        logging.warning('FIRST_PEAK: No peaks found, returning empty peak information')
        return {'peak': 0, 'left': 0, 'right': 0} 
    
    if peaks[1]['left_bases'][0] == 0 and len(peaks[0]) > 1:
        peak_index = 1
    elif peaks[1]['left_bases'][0] == 0 and len(peaks[0]) == 1:
        peak_index = 0
        logging.warning('FIRST_PEAK: Signal only contains one peak, which starts at 0')
    else:
        peak_index = 0
        
    peak_location = peaks[0][peak_index]
    left_base = peaks[1]['left_bases'][peak_index]
    right_base = peaks[1]['right_bases'][peak_index]
    return {'peak': peak_location, 'left': left_base, 'right': right_base}

def get_half_max_up(signal, peak):
    """
    Estimate the time to half max signal.
    This is a linear interpolation based on three points surrounding the
    point with the value closest to the half-max value. Thus it is fairly
    inexact. Consider replacing by modeling the peak rise and fitting the
    model to the data.

    Args
    ----
        signal: 1D numpy array
        peak: a dict that is the output of `first_good_peak()`.

    Returns
    -------
        half_max_up: floating point number
    """
    # If no peaks, return nothing
    if peak['peak'] == 0:
        return np.nan
    
    fflag = False # does this method fail?
    # Take half the value of the signal at the peak index
    half_max = signal[peak['peak']] / 2

    # Take only the part of the signal relevant to half_max_up calculation
    # i.e., only the signal between the left base of the first peak and the
    # max of the first peak
    rising_signal = signal[peak['left']:(peak['peak']+1)]

    # Find the index of the signal sample closest to half_max
    closest_idx = (np.abs(rising_signal - half_max)).argmin() + peak['left']
    if closest_idx <= 1 or closest_idx >= 98:
        logging.warning('HM_UP: half-max too close to end of signal')
        return np.nan

    # If the signal at the index is nearly equal to half max, take that index 
    if np.allclose(half_max, signal[closest_idx]):
        half_max_point = closest_idx

    # ...otherwise interpolate
    else:
        ix = -1
        triplet = signal[(closest_idx - 1):(closest_idx + 2)]
        if triplet[0] < half_max < triplet[1]:
            ix = 0
        elif triplet[1] < half_max < triplet[2]:
            ix = 1
        else:
            logging.warning('HM_UP: simple method for interpolating half-max'
                            ' rise time failed')
            fflag = True
        if ix != -1:
            y = [ix,ix+1]
            x = [triplet[ix], triplet[ix+1]]
            f = interp1d(x,y)
            trip_coord = f(half_max)
            half_max_point = closest_idx + (trip_coord - 1)
    if fflag == True:
        half_max_up = np.nan
    else:
        half_max_up = float(half_max_point - peak['left'])
    return half_max_up

# The next two functions are to break the peak values out into new 
# data frame columns
def get_peak_values(signal, peak, pad=3):
    left_ix = np.max([peak['left'] - pad, 0])
    right_ix = np.min([peak['right'] + pad, 99])
    peak_values = signal[left_ix:right_ix+1]
    return peak_values

def create_peak_value_df(peak_values):
    columns_needed = np.array([n.size for n in peak_values]).max()
    arr = np.zeros([len(peak_values), columns_needed])
    for r,cols in enumerate(peak_values):
        arr[r, 0:len(cols)] = cols
    colnames = [f'p{i}' for i in range(0,columns_needed)]
    peak_value_df = pd.DataFrame(arr, columns = colnames)
    return peak_value_df

def get_half_max_down(signal, peak):
    """See `get_half_max_up` for explanation.

    This is a minor modification of the above function.
    """
    if peak['peak'] == 0:
        return np.nan
    fflag = False
    half_max = signal[peak['peak']] / 2
    falling_signal = signal[peak['peak']:(peak['right']+1)]
    closest_idx = (np.abs(falling_signal - half_max)).argmin() + peak['peak']

    if closest_idx <= 1 or closest_idx >= 98:
        logging.warning('HM_DOWN: half-max too close to end of signal')
        return np.nan

    # If the signal at the index is nearly equal to half max, take that index 
    if np.allclose(half_max, signal[closest_idx]):
        half_max_point = closest_idx

    # ...otherwise interpolate
    else:
        ix = -1
        triplet = signal[(closest_idx - 1):(closest_idx + 2)]
        if triplet[0] > half_max > triplet[1]:
            ix = 0
        elif triplet[1] > half_max > triplet[2]:
            ix = 1
        else:
            logging.warning('HM_DOWN: simple method for interpolating'
                            ' half-max decay time failed')
            fflag = True
        if ix != -1:
            y = [ix,ix+1]
            x = [triplet[ix], triplet[ix+1]]
            f = interp1d(x,y)
            trip_coord = f(half_max)
            half_max_point = closest_idx + (trip_coord - 1)
    if fflag == True:
        half_max_down = np.nan
    else:
        half_max_down = float(half_max_point - peak['peak'])
    return half_max_down

def flip_signal(signal, flag):
    if flag:
        signal = -(signal - signal.max())
    return signal

def main(fp, df_flag, c_flag):
    if df_flag == c_flag:
        logging.error("You must use EITHER the --deltaf OR --contraction flag;"
                      " Doing nothing." )
        return

    try:
        df = pd.read_excel(fp)
    except FileNotFoundError:
        logging.error(f"No file found called \"{fp}\", try using absolute path"
                        "or checking your working directory")
        return
    outfn = ".".join(basename(fp).split('.')[:-1]) + '_processed.xlsx'

    try:
        signals = df.loc[:, 't0':'t99'] #consider making this configurable
    except:
        logging.error("Spreadsheet is improperly formatted; doing nothing.")
        return

    smoothed = signals.apply(smooth_signal, axis=1, window_size=3)

    if c_flag:
        try:
            orientations = df.Replicate
        except:
            logging.error("Replicate column required for contraction data;"
                          " doing nothing.")
            return

        orientations = orientations.astype('category')
        if set(['Bottom', 'Top']) != set(orientations.dtypes.categories):
            logging.error("Replicate values must be in ['Bottom', 'Top'];"
                          " doing nothing.")
            return
        
        #flip the smoothed signals if necessary
        for idx, orientation in enumerate(orientations):
            if orientation == 'Bottom':
                smoothed[idx] = -(smoothed[idx] - smoothed[idx].max())

    baselines = smoothed.apply(create_baseline)

    if df_flag:
        proc_sig = pd.Series([deltaf(series, baseline)
                              for series, baseline
                              in zip(smoothed, baselines)])

    else:
        proc_sig = pd.Series([(series - baseline)
                              for series, baseline
                              in zip(smoothed, baselines)])

    norm_sig = [normalize_df(series) for series in proc_sig]
    peaks = [find_peaks(series, prominence=0.2, wlen=35) for series in norm_sig]
    target_peaks = [first_good_peak(p) for p in peaks]

    peak_magnitude = []
    peak_duration = []
    rise = []
    decay = []
    area_under_peak = []
    half_max_up = []
    peak_values = []
    half_max_down = []
    for peak, signal in zip(target_peaks, proc_sig):
        peak_magnitude.append(signal[peak['peak']])
        peak_duration.append(peak['right'] - peak['left'])
        rise.append(peak['peak'] - peak['left'])
        decay.append(peak['right'] - peak['peak'])
        area_under_peak.append(signal[peak['left']:(peak['right']+1)].sum())
        half_max_up.append(get_half_max_up(signal, peak))
        peak_values.append(get_peak_values(signal, peak))
        half_max_down.append(get_half_max_down(signal, peak))

    peak_value_df = create_peak_value_df(peak_values)

    output_df = df.assign(peak_magnitude=peak_magnitude,
                          peak_duration=peak_duration,
                          rise=rise,
                          decay=decay,
                          area_under_peak=area_under_peak,
                          half_max_up=half_max_up,
                          half_max_down=half_max_down)
    output_df = output_df.join(peak_value_df)

    with pd.ExcelWriter(outfn) as writer:
        output_df.to_excel(writer)
        logging.info(f'Writing output file as {outfn}')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Command line help and argument parsing
    description = ("Process xlsx files containing Hinson lab calcium data, "
                   "see script header for details")
    parser = ArgumentParser(description=description)
    parser.add_argument('file', type=str,
                        help='Path to the *.xlsx file containing signal data')
    parser.add_argument('--deltaf', action='store_true',
                        help='Flag to indicate deltaf/f should be calculated')
    parser.add_argument('--contraction', action='store_true',
                        help='Flag to indicate contraction data is to be'
                             'processed')
    args = parser.parse_args()

    main(fp=args.file, df_flag=args.deltaf, c_flag=args.contraction)
