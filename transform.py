# Standard libraries
import pandas as pd

# Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper

# Logging
import logging
logging.basicConfig(level=logging.INFO)

def transform(path_to_csv):
    """
    Function that takes in a path to
    a dataframe and transforms it to
    be model-ready by re-ordering the
    columns and returns non-normalized
    and normalized versions of the df

    :param path_to_csv: str

    :return: 2 dfs (1 non-normalized,
    1 normalized): pandas.DataFrame,
    pandas.Dataframe
    """

    # Read in data as pandas dataframe
    data = pd.read_csv(path_to_csv)

    # Check class distribution (proportion of minor vs. major)
    logging.info('Class distribution:')
    logging.info('minor: {}%'.format(round((data['mode'].value_counts()[0] / data.shape[0]), 2) * 100))
    logging.info('major: {}%'.format(round((data['mode'].value_counts()[1] / data.shape[0]), 2) * 100))

    print(data.columns)

    # Reorder columns
    data = data[['tonic_note',
                 'BPM',
                 'rmse_mean',
                 'rmse_std',
                 'spec_cent_mean',
                 'spec_cent_std',
                 'zcr_mean',
                 'zcr_std',
                 'tonic_int',
                 'mode_int'
                 ]]

    # Normalize data
    # Create mapper with StandardScaler()
    mapper = DataFrameMapper([
        (['BPM'], MinMaxScaler(feature_range=(0, 1))),
        (['rmse_mean'], MinMaxScaler(feature_range=(0, 1))),
        (['rmse_std'], MinMaxScaler(feature_range=(0, 1))),
        (['spec_cent_mean'], MinMaxScaler(feature_range=(0, 1))),
        (['spec_cent_std'], MinMaxScaler(feature_range=(0, 1))),
        (['zcr_mean'], MinMaxScaler(feature_range=(0, 1))),
        (['zcr_std'], MinMaxScaler(feature_range=(0, 1))),
        (['tonic_int'], MinMaxScaler(feature_range=(0, 1))),
        (['mode_int'], MinMaxScaler(feature_range=(0, 1)))

    ], df_out=True)

    # Fit and transform mapper
    data_norm = mapper.fit_transform(data.copy())

    return(data, data_norm)