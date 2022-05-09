# ##############################################################################
# Usage: python run_PCA_Kmeans.py {csv_path} start_column end_column
# python run_PCA_Kmeans.py
#               sample_input/ENV18PM_Pro_QCT_subj177_20211008.csv
#               dAV_U_ML_T0
#               Dhn_sRLL_IN0
#               save
# Run Time: ~ 20s
# Ref: [Saccenti E et al., 2015 chemolab], [Horn J, 1965 psychometrika]
# ##############################################################################
# 20211008, In Kyu Lee
# Desc: Run PCA
#      - number of PC is determined by Horn's parallel analysis
# ##############################################################################
# Input:
#  - csv data file path
#  - first column of variable to run PCA, usually after Subj column
#       ex)'Age_yr'
#  - last column of variable to run PCA, usually the last column
#       ex) 'Dhn_sRLL_IN0'
# Output:
#  - PC components csv file
#    - save in the same path to the {csv_path} with _Kmeans_n.csv suffix.
# ##############################################################################
# More combinations 
# QCT only: dAV_U_ML_T0 Dhn_sRLL_IN0
# QCT + demo: Age_yr Dhn_sRLL_IN0
# ##############################################################################
import os
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from factor_analyzer import FactorAnalyzer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS


def plot_PCA(df):
    n_var = df.shape[1]
    pca = PCA().fit(df)
    plt.rcParams["figure.figsize"] = (15, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, n_var + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker="o", linestyle="--", color="b")
    plt.xlabel("Number of Components")
    plt.xticks(
        np.arange(0, n_var + 1, step=1)
    )  # change from 0-based array index to 1-based human-readable label
    plt.ylabel("Cumulative variance (%)")
    plt.title("The number of components needed to explain variance")
    plt.axhline(y=0.95, color="r", linestyle="-")
    # plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
    ax.grid(axis="x")
    plt.show()


def run_PCA(df, n):
    pca = PCA(n_components=n)
    pcs = pca.fit_transform(df)
    weights = pca.components_
    columns = [f"PC{i+1}" for i in range(n)]
    pc_df = pd.DataFrame(data=pcs, columns=columns)
    return pc_df, weights


def impute_df(df, strategy="mean"):
    # strategy = ['mean','median',most_frequent','constant']
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    new_df = imp.fit_transform(df)
    new_df = pd.DataFrame(new_df, columns=df.columns.values)
    return new_df


def _HornParallelAnalysis(data, K=10, printEigenvalues=False, show_plot=True):
    # Create a random matrix to match the dataset
    n, m = data.shape
    # Set the factor analysis parameters
    fa = FactorAnalyzer(n_factors=1, method="minres", rotation=None, use_smc=True)
    # Create arrays to store the values
    sumComponentEigens = np.empty(m)
    # sumFactorEigens = np.empty(m)
    # Run the fit 'K' times over a random matrix
    for runNum in range(0, K):
        fa.fit(np.random.normal(size=(n, m)))
        sumComponentEigens = sumComponentEigens + fa.get_eigenvalues()[0]
        # sumFactorEigens = sumFactorEigens + fa.get_eigenvalues()[1]
    # Average over the number of runs
    avgComponentEigens = sumComponentEigens / K
    # avgFactorEigens = sumFactorEigens / K

    # Get the eigenvalues for the fit on supplied data
    fa.fit(data)
    dataEv = fa.get_eigenvalues()
    # Set up a scree plot

    # Print results
    if printEigenvalues:
        print(
            "Principal component eigenvalues for random matrix:\n", avgComponentEigens
        )
        # print('Factor eigenvalues for random matrix:\n', avgFactorEigens)
        print("Principal component eigenvalues for data:\n", dataEv[0])
        # print('Factor eigenvalues for data:\n', dataEv[1])
    # Find the suggested stopping points
    # suggestedFactors = sum((dataEv[1] - avgFactorEigens) > 0)
    suggestedComponents = sum((dataEv[0] - avgComponentEigens) > 0)
    print("Parallel analysis suggests the number of components = ", suggestedComponents)

    if show_plot:
        plt.figure(figsize=(15, 6))
        # For the random data - Components
        plt.plot(
            range(1, m + 1), avgComponentEigens, "b", label="PC - random", alpha=0.4
        )
        # For the Data - Components
        plt.scatter(range(1, m + 1), dataEv[0], c="b", marker="o")
        plt.plot(range(1, m + 1), dataEv[0], "b", label="PC - data")
        # For the random data - Factors
        # plt.plot(range(1, m+1), avgFactorEigens, 'g', label='FA - random', alpha=0.4)
        # For the Data - Factors
        # plt.scatter(range(1, m+1), dataEv[1], c='g', marker='o')
        # plt.plot(range(1, m+1), dataEv[1], 'g', label='FA - data')
        plt.title("Parallel Analysis Scree Plots", {"fontsize": 20})
        plt.xlabel("Principal Components", {"fontsize": 15})
        plt.xticks(ticks=range(1, m + 1), labels=range(1, m + 1))
        plt.ylabel("Eigenvalue", {"fontsize": 15})
        plt.legend()
        plt.show()
    return suggestedComponents


def run_Kmeans(df, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    labels = kmeans.predict(df)
    return labels


def main():
    # df_path = 'sample_input/20180829_ClusterAnalysisDataTable.csv'
    # df_path = "sample_input/ENV18PM_IN0_EX0_QCT_all_20211008_176subjs_PCA7.csv"
    # ---------------------------
    df_path = str(sys.argv[1])
    first_var = str(sys.argv[2])
    last_var = str(sys.argv[3])
    SAVE = str(sys.argv[4])
    # ---------------------------
    raw_df = pd.read_csv(df_path)
    # df = raw_df.iloc[:, 6:]
    df = raw_df.loc[:, first_var:last_var]
    print("Running PCA & K-means on: ")
    print(df.columns)

    # Check missing values
    if df.isnull().values.any():
        df = impute_df(df)
    # normalize before PCA
    scaler = MinMaxScaler()
    df_rescaled = scaler.fit_transform(df)

    # PCA #
    PCA_n = _HornParallelAnalysis(df_rescaled, show_plot=False)
    pc_df, weights = run_PCA(df_rescaled, PCA_n)
    PC_index = [f'PC{i}' for i in range(PCA_n)]
    PC_weights = pd.DataFrame(weights, columns=df.columns.values,index=PC_index)
    PC_weights = PC_weights.astype('float16')
    print("-------------------------------------")
    print(PC_weights)
    print("-------------------------------------")
    # K-Means #
    best_score = 0
    print("-------------------------------------")
    for n in range(2, 6):
        labels = run_Kmeans(pc_df, n)
        score = silhouette_score(pc_df, labels)
        print(f"Silhouette score of {n} cluster:{score:.3f}")
        if best_score < score:
            best_score = score
            best_k = n
    print("-------------------------------------")
    k = int(input("Enter K: "))
    print(f"Run K-means with k={k}")
    labels = run_Kmeans(pc_df, k)

    # Save cluster info
    if SAVE == 'save':
        raw_df["Cluster"] = labels
        kmeans_file = df_path.split("/")[-1][:-4] + f"_{first_var}_{last_var}_Kmeans_{k}.csv"
        kmeans_path = os.path.join("".join(df_path.split("/")[:-1]), kmeans_file)
        PCA_file = df_path.split("/")[-1][:-4] + f"_{first_var}_{last_var}_PCA{PCA_n}_weights.csv"
        PCA_path = os.path.join("".join(df_path.split("/")[:-1]), PCA_file)
        raw_df.to_csv(kmeans_path, index=False)
        PC_weights.to_csv(PCA_path, index=False)
        print("-------------------------------------")
        print("Result PCA_weight file is saved as:")
        print(PCA_path)
        print("-------------------------------------")
        print("Result K_means file is saved as:")
        print(kmeans_path)
        print("-------------------------------------")


if __name__ == "__main__":
    main()
