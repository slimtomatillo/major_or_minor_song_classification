# Standard libraries
import numpy as np
import pandas as pd

# OS
import os
from os import listdir
from os.path import isfile
from os.path import join

# Audio processing
import scipy.linalg
import scipy.stats
import librosa as lb
import librosa.display
from pydub import AudioSegment

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# This file contains 3 functions:

# 1. extract()
    # Calls get_features()

# 2. get_features()
    # Calls ks_key()

# 3. ks_key()

def extract(path_to_songs_string, clip_length_in_seconds, path_for_sliced_clips):

    """
    Function that takes in the path to a
    folder containing music files, extracts
    features from clip of song (length
    of clip = parameter), and returns df

    Also counts the number of songs
    processed.

    :param path_to_songs_string: str
    :param clip_length_in_seconds: int
    :param path_for_sliced_clips: str

    :return: pandas df of extracted
    features: pandas.DataFrame
    """

    logging.info('Collecting paths to all audio files in specified folder...')

    # Make sure input paths are strings
    path_to_songs_string = str(path_to_songs_string)
    path_for_sliced_clips = str(path_for_sliced_clips)

    # List of all song file paths
    list_of_paths = []

    # Lists all files in the given directory
    allfiles = os.listdir(path_to_songs_string)

    # Iterate over all files in given directory
    for item in allfiles:

        # If item is mp3 file, add the path to the list
        if item.endswith('.mp3'):
            list_of_paths.append(path_to_songs_string + '/' + str(item))

        # If item is m4a file, add the path to the list
        if item.endswith('.m4a'):
            list_of_paths.append(path_to_songs_string + '/' + str(item))

    # Song counter
    file_count = len(list_of_paths)

    logging.info('Songs found: {}'.format(file_count))
    logging.info('Finished collecting audio file paths.')
    logging.info('Splitting audio files to specified length...')

    d_count = 1

    # Go through each song (to split)
    for song_path in sorted(list_of_paths):
        logging.info(song_path.split('/')[-1])
        logging.info(d_count)
        d_count +=1

        # Create clip file name
        name = song_path.split('/')[-1].split('.')[0]

        # Open mp3 using pydub's AudioSegment class
        mp3 = AudioSegment.from_file(song_path)

        # Get clip of size specified and export
        splice_length = clip_length_in_seconds * 1000
        clip = mp3[:splice_length]

        # Export clip
        with open(path_for_sliced_clips + '/' + '{}.mp3'.format(name), "wb") as f:
            clip.export(f, format='wav')

    logging.info('Finished splicing songs.')

    # Get list of all clips in folder
    clip_names = [f for f in listdir(path_for_sliced_clips) if isfile(join(path_for_sliced_clips, f))]

    # Consolidate paths of all songs
    all_clip_paths = [(path_for_sliced_clips + '/' + x) for x in clip_names]

    # Get rid of the hidden file containing attributes of the folder called '.DS_Store'
    all_clip_paths = [c for c in all_clip_paths if '.DS_Store' not in c]

    # Sort list of clip paths
    all_clip_paths.sort()

    logging.info('Total number of clips: {}'.format(len(all_clip_paths)))
    logging.info('Beginning feature extraction')

    # Create list that will contain each dict per clip
    clip_features = []

    # Get features for each song clip and put in dict
    num = 1
    for clip in all_clip_paths:
        logging.info('Extracting features from clip: {}'.format(num))
        #print(clip.split('/')[-1].split('.')[0])
        curr_dict = get_features(clip)
        clip_features.append(curr_dict)
        num += 1

    logging.info('Finished feature extraction.')
    logging.info('Converting data to df.')

    # Convert the list of dicts to dataframe
    data = pd.DataFrame(data=clip_features)

    return(data)

def get_features(song_filepath_string):
    """
    Function that takes in the path to
    a song and extracts audio features,
    returning a dictionary of features
    for that song

    :param song_filepath_string: str

    :return: dict of song features: dict
    """

    # Get the file path
    filename = str(song_filepath_string)

    # Get song name
    name = filename.split('/')[-1].split('.')[0]

    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr`
    y, sr = librosa.load(filename, sr=22050)

    # Run the default beat tracker, get tempo
    tempo, beat_frames = lb.beat.beat_track(y=y, sr=sr)

    # Set the hop length
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = lb.effects.hpss(y)

    # Compute MFCC features from the raw signal
    mfcc = lb.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = lb.feature.delta(mfcc)

    # Stack and synchronize between beat events
    beat_mfcc_delta = lb.util.sync(np.vstack([mfcc, mfcc_delta]),
                                   beat_frames
                                   )

    # Compute chroma features from the harmonic signal
    chromagram = lb.feature.chroma_cqt(y=y_harmonic,
                                       sr=sr
                                       )

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = lb.util.sync(chromagram,
                               beat_frames,
                               aggregate=np.median
                               )

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

    # Get scores of every pitch throughout song from chromagram and ks_key function
    major_minor_scores = ks_key(chromagram)

    major_scores = major_minor_scores[0]
    minor_scores = major_minor_scores[1]

    # Determine dominant note of key
    highest = []
    for x in range(0, len(major_scores[0])):
        i = np.argmax(major_scores[:, x])
        highest.append(i)

    # Create dict of numbers corresponding to pitches
    pitch_dict = {0: 'C',
                  1: 'C# / Db',
                  2: 'D',
                  3: 'D# / Eb',
                  4: 'E',
                  5: 'F',
                  6: 'F# / Gb',
                  7: 'G',
                  8: 'G# / Ab',
                  9: 'A',
                  10: 'A# / Bb',
                  11: 'B'}

    # Get mode (major or minor)
    highest_count = 0
    top_num = 0

    for x in range(0, 12):
        curr_count = highest.count(x)
        if curr_count > highest_count:
            highest_count = curr_count
            top_num = x

    # Get number representing base note
    tonic = top_num

    # Major third follows pattern of 2 whole steps
    major_third = tonic + 4

    # Minor third follows pattern of whole step, 1/2 step
    minor_third = tonic + 3

    # Find which third (major or minor) appears more
    if highest.count(major_third) > highest.count(minor_third):
        the_mode = 'major'
        mode_int = 1
    else:
        the_mode = 'minor'
        mode_int = 2

    # Get dominant note
    tonic_note = pitch_dict[tonic]

    # Get spectral centroid
    spectral_centroids_adj = lb.feature.spectral_centroid(y + 0.01, sr=sr)[0]
    sc_mean = spectral_centroids_adj.mean().round(2)
    sc_std = spectral_centroids_adj.std().round(2)

    # Get zero crossing rate
    zero_cr = lb.feature.zero_crossing_rate(y)
    zero_cr_mean = lb.feature.zero_crossing_rate(y).mean().round(4)
    zero_cr_std = lb.feature.zero_crossing_rate(y).std().round(4)

    # Get root-mean-square energy
    rmse = lb.feature.rmse(y=y)
    rmse_mean = rmse.mean().round(4)
    rmse_std = rmse.std().round(4)

    # Create dict of song features
    song_features = {'title': name,
                     'BPM': int(round(tempo, 2)),
                     'tonic_note': tonic_note,
                     'tonic_int': tonic,
                     'mode': the_mode,
                     'mode_int': mode_int,
                     'spec_cent_mean': sc_mean,
                     'spec_cent_std': sc_std,
                     'zcr_mean': zero_cr_mean,
                     'zcr_std': zero_cr_std,
                     'rmse_mean': rmse_mean,
                     'rmse_std': rmse_std
                     }

    return(song_features)


def ks_key(X):
    """
    Function that estimates the key from a
    pitch class distribution

    :param X: pitch-class energy distribution:
    np.ndarray

    :return: 2 arrays of correlation scores for
    major and minor keys: np.ndarray (shape=(12,)),
    np.ndarray (shape=(12,))
    """
    X = scipy.stats.zscore(X)

    # Coefficients from Kumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    major = scipy.stats.zscore(major)

    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    minor = scipy.stats.zscore(minor)

    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)

    return(major.T.dot(X), minor.T.dot(X))
