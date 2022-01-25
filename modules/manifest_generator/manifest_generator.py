from sklearn.model_selection import train_test_split

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram_bins(x):
    ax = x.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(10, 7))
    # ax.set_xticklabels([c[1:-1].replace(","," to") for c in x.cat.categories])
    plt.show()


def get_bins(dataset, bins=range(0, 100, 10), plot=False):

    manual_binned_data = pd.cut(dataset["length"], bins=bins)
    quantile_binned_date, bins = pd.qcut(dataset["length"], q=len(bins),
                                         retbins=True)  # Generates 10 bins (same as manual)

    print(f"Ranges for quantiles calculated: {bins}")

    if plot:
        plot_histogram_bins(manual_binned_data)
        plot_histogram_bins(quantile_binned_date)

    dataset["manual_bin"] = manual_binned_data
    dataset["quantile_bin"] = quantile_binned_date

    return dataset


def split_dataset(dataset, stratify_col, plot=False):

    X_train, X_val = train_test_split(dataset, test_size=0.3, stratify=dataset[stratify_col], random_state=42)

    if plot:
        plot_histogram_bins(X_train[stratify_col])
        plot_histogram_bins(X_val[stratify_col])

    return X_train, X_val


def generateTrainValPartitions(file):

    DATASET_FOLDER = os.getcwd() + "/{0}/".format(file)



    PATH_TO_MANIFEST = DATASET_FOLDER + "dataset_labels.csv"
    if os.path.exists(PATH_TO_MANIFEST):
        manifest_df = pd.read_csv(PATH_TO_MANIFEST)

        manifest_df["length"] = manifest_df["length"].astype(float)  # To make sure that the length is not in string format

        # Bins the dataset
        binned_dataset = get_bins(manifest_df, bins=range(0, 100, 10), plot=False)

        # Splits the dataset based on the manual bins generation
        X_train, X_val = split_dataset(binned_dataset, "manual_bin")

        # Drops generated columns
        X_train = X_train.drop(columns=["manual_bin", "quantile_bin"])
        X_val = X_val.drop(columns=["manual_bin", "quantile_bin"])

        # Dumps the manifest files
        X_train.to_csv(DATASET_FOLDER + "train.csv", index=False)
        X_val.to_csv(DATASET_FOLDER + "val.csv", index=False)
    else:
        print("The dataset folder: {0} does not contains the file 'dataset_labels.csv'".format(DATASET_FOLDER))