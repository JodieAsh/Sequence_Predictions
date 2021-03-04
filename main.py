import argparse
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import pickle
import more_itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import R packages
utils = rpackages.importr('utils')
base = rpackages.importr('base')
# Select a mirror for R packages
utils.chooseCRANmirror(graphics=False, ind=1)  # select the first mirror in the list
# Install R packages
devtools = utils.install_packages('devtools')

# Install the Epitopes R package
d = {'package.dependencies': 'package_dot_dependencies',
     'package_dependencies': 'package_uscore_dependencies'}
custom_analytics = rpackages.importr('devtools', robject_translations=d)
epitopes = custom_analytics.install_github("fcampelo/epitopes")
prepfeatgen = custom_analytics.install_github("JodieAsh/FeatGenPrep")
# Load packages
prepfeatgen = rpackages.importr('FeatGenPrep')
epitopes = rpackages.importr('epitopes')


def window_sequence(sequence, n=15, step=1):
    """Using the more_itertools package.
    Returns a sliding window (of width n) over the sequence."""
    results = list(more_itertools.windowed(sequence, n=n, step=step))

    windowed_data = []
    for window in results:
        windowed_seq = ""
        for letter in window:
            windowed_seq += letter
        windowed_data.append(windowed_seq)

    return windowed_data


def remove_info_cols(dataset):
    """
    Removes any columns with the prefix "info" from a given dataset.
    :param dataset: Pandas.DataFrame
    Dataset to remove "info" columns from.
    :return: Pandas.DataFrame
    Dataset with "info" columns removed.
    """
    new_cols = [c for c in dataset.columns if c.lower()[:4] != "info"]
    dataset = dataset[new_cols]
    return dataset


def normalise_data(data):
    """
    Standardize features by removing the mean and scaling to unit variance (using Scikit-learn).
    :param data: Pandas.DataFrame
    Dataset to be normalised.
    :return: Pandas.DataFrame
    Dataset that has been normalised.
    """
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data


def main(sequence, model):
    # Split the sequence into 15 amino acid long windows
    windowed_data = window_sequence(sequence)
    # Convert to Pandas DataFrame
    data = pd.DataFrame(data=windowed_data)
    # Rename column
    data.columns = ['Info_window_seq']
    # Add dummy column as R conversion does not work for 1D dataframes
    data['Info_dummy'] = 0

    # Convert data for use with R packages
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)
    # Generate features for the windowed sequences
    r_data = prepfeatgen.prepare_columns(r_data)
    r_data = epitopes.calc_features(r_data)
    # Convert R data frame back to Pandas
    with localconverter(ro.default_converter + pandas2ri.converter):
        data = ro.conversion.rpy2py(r_data)
    print('\n')

    # Pre-process the data before classification
    data = remove_info_cols(data)
    data = normalise_data(data)

    # Load the model to make predictions with
    if model == "Onchocerca volvulus":
        model = pickle.load(open(r"Models\O.volvulus_training_data_benchmarks_RF.pkl", 'rb'))
    if model == "Epstein-Barr Virus":
        model = pickle.load(open(r"Models\Epstein-Barr_Virus_training_benchmarks_RF.pkl", 'rb'))
    if model == "Hepatitis C Virus":
        model = pickle.load(open(r"Models\Hepatitis_C_Virus_training_benchmarks_RF.pkl", 'rb'))
    # Make predictions on the window sequences (features)
    predictions = model.predict(data)

    # Add predictions next to sequences in data frame
    results = pd.DataFrame(data=windowed_data)
    results.columns = ['Window_sequence']
    results['Predictions'] = predictions
    # Add position
    results['Position'] = range(1, len(results) + 1)
    # Convert / save to csv file
    results.to_csv("results.csv", index=False)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', '-s', type=str, required=True, help='Protein sequence to make predictions on.')
    parser.add_argument('--model', '-m', type=str, required=True, help='Name of model to run protein sequence through.')

    args = parser.parse_args()

    main(sequence=args.sequence, model=args.model)


# example_sequence = "MFYYLGLLVMIVFILQAIAFLVLLERHFLGGSQCRVGPNKVGYCGVLQALFDGLKLLKKEQLLLCFSSWLSFLFMPICGFVLMVFFWFTLPYFFSE"
