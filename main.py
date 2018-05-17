# Functions
from extract import extract
from model import model

import os

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Set directory
dir = '/Users/alexandrasmith/ds/metis/proj3_mcnulty/PROJ_FILES/major_or_minor_song_classification'

# filepath_to_music --> Edit as needed
filepath = 'sample_music_files'

# desired_clip_length --> Edit as needed (suggested: 30)
clip_sec = 30

# path_for_sliced_clips --> Edit as needed
export_path = 'sample_music_files_sliced'

def main(filepath_to_music, desired_clip_length, path_for_sliced_clips):
    """
    Function that takes in music files,
    extracts audio features, creates
    pandas df, transforms data for
    modeling, builds models, stores
    performance of models

    Takes in (1) a filepath to where the
    music is stored, (2) the desired clip
    length to build the models on, and (3)
    the path to store the sliced clips
    (which can be deleted after csv is
    exported

    :param filepath_to_music: str
    :param desired_clip_length: int
    :param path_for_sliced_clips: str

    :return: None, exported df as csv,
    exported df of models and performance
    """
    global dir

    logging.info('Extracting features...')

    # Call extract() on filepath_to_music
    data = extract(filepath_to_music, desired_clip_length, path_for_sliced_clips)

    logging.info('Done with extraction.')

    # Check if folder exists, if not, make it
    if not os.path.exists(dir):
        os.mkdir(dir)

    # Export dataframe as csv
    data.to_csv(os.path.join(dir, f'{desired_clip_length}_sec_data.csv'), index=False)

    logging.info('Data exported to csv.')
    logging.info('Building models...')

    # Call model on extracted files
    model_perf, holdout_perf = model(os.path.join(dir, f'{desired_clip_length}_sec_data.csv'))

    # Export dataframes of performance as csv
    model_perf.to_csv(os.path.join(dir, f'{desired_clip_length}_sec_performance.csv'), index=False)
    holdout_perf.to_csv(os.path.join(dir, f'{desired_clip_length}_sec_holdout.csv'), index=False)

    logging.info('Finished.')

    return

if __name__ == '__main__':
    main(filepath, clip_sec, export_path)
