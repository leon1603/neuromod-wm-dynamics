"""
Multicollinearity_Analysis.py
------------------------------
Multicollinearity analysis for receptor densities and/or T1w/T2w myelination
data, using Variance Inflation Factors (VIF) decomposed at global, within-system,
and cross-system levels, plus a Pearson correlation heatmap.

Public API
----------
- calculate_multicollinearity()
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

from Scripts.utils import SYSTEM_MAPPING, ALL_RECEPTOR_COLUMNS

def load_and_select_features(
    X_path: str,
    selected_receptors: list = None,
) -> tuple:
    """
    Load feature data from CSV, optionally filter to a receptor subset, and z-score normalise.

    Parameters
    ----------
    X_path : str
        Path to CSV with one column per receptor/feature and an optional 'ROI' index column.
    selected_receptors : list of str or None
        Features to retain. If None, all columns are used.

    Returns
    -------
    df_scaled : DataFrame, z-scored, columns = selected features, index = ROI names.
    available : list of str, the features actually present after filtering.
    """
    header = pd.read_csv(X_path, nrows=0).columns  # read only the header row to check whether an 'ROI' column exists
    df = pd.read_csv(X_path, index_col="ROI" if "ROI" in header else None)  # load the full CSV, setting 'ROI' as the index if present so rows are labelled by brain region

    if selected_receptors is None:
        available = df.columns.tolist()  # if no subset was requested, use every column in the file
    else:
        available = [r for r in selected_receptors if r in df.columns]  # retain only receptors that were both requested and actually present in the data

    df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(df[available]),  # z-score each feature column (subtract mean, divide by std) so all receptors are on a comparable scale
        index=df.index,    # preserve the original ROI row labels
        columns=available, # preserve the selected feature column names
    )

    return df_scaled, available  # return the scaled DataFrame and the list of features that survived filtering


def build_output_suffix(available: list) -> str:
    """
    Derive a directory name and filename suffix that reflects which receptors are included.

    Rules
    -----
    - All receptors + T1w:T2w   → 'AllReceptors_T1wT2w'
    - All receptors, no T1w:T2w → 'AllReceptors'
    - Any other subset           → '<receptor1>_<receptor2>_...'

    Parameters
    ----------
    available : list of str

    Returns
    -------
    suffix : str
    """
    ALL_RECEPTOR_NAMES = set(ALL_RECEPTOR_COLUMNS)  # convert to a set for fast membership testing

    selected_set = set(available)                           # the set of features actually in this run
    has_t1wt2w = "T1w:T2w" in selected_set                 # True if T1w:T2w was included
    receptors_only = selected_set - {"T1w:T2w"}            # remove T1w:T2w to isolate just the receptor names
    all_receptors_used = ALL_RECEPTOR_NAMES.issubset(receptors_only)  # True only if every receptor is present

    if all_receptors_used and has_t1wt2w:
        return "AllReceptors_T1wT2w"  # all 19 receptors plus myelination. use this descriptive label
    elif all_receptors_used:
        return "AllReceptors"          # all 19 receptors but no myelination
    else:
        sanitized = [r.replace(":", "") for r in available]  # remove colons from names like 'T1w:T2w' so the suffix is filesystem-safe
        return "_".join(sanitized)     # join feature names with underscores to form a unique, readable label


def build_filenames(suffix: str) -> dict:
    """
    Return a dict of output filenames keyed by content type.

    Parameters
    ----------
    suffix : str, as returned by build_output_suffix().

    Returns
    -------
    filenames : dict with keys 'system', 'receptor', 'corr_matrix', 'heatmap'.
    """
    return {
        "system":      f"System_Multicollinearity_{suffix}.csv",      # global condition number summary
        "receptor":    f"Receptor_Multicollinearity_{suffix}.csv",     # per-receptor global/within-system/cross-system VIFs
        "corr_matrix": f"Receptor_Correlation_Matrix_{suffix}.csv",   # full Pearson correlation matrix
        "heatmap":     f"Receptor_Correlation_Heatmap_{suffix}.png",  # visual heatmap of the correlation matrix
    }


def compute_vif_for_predictor(
    target: str,
    predictors: list,
    df_scaled: pd.DataFrame,
) -> float:
    """
    Compute the Variance Inflation Factor for one feature given a set of predictors.

    Parameters
    ----------
    target     : str, the feature whose VIF is being computed.
    predictors : list of str, the features used as regressors.
    df_scaled  : DataFrame containing all scaled features.

    Returns
    -------
    vif : float
    """
    if len(predictors) == 0:
        return 1.0  # a feature with no other predictors has no collinearity, so VIF is exactly 1

    model = OLS(df_scaled[target], add_constant(df_scaled[predictors])).fit()  # regress the target receptor on the predictor set to measure how well it can be linearly predicted by the others
    return 1.0 / (1.0 - model.rsquared) if model.rsquared < 1.0 else np.inf  # VIF = 1/(1-R²)


def compute_global_vifs(df_scaled: pd.DataFrame, available: list) -> tuple:
    """
    Compute global VIFs for all features simultaneously using statsmodels.

    Parameters
    ----------
    df_scaled : DataFrame of z-scored features.
    available : list of str, feature column names.

    Returns
    -------
    global_vifs : list of float, one value per column (index 0 = intercept).
    X_global    : DataFrame with constant column prepended.
    """
    X_global = add_constant(df_scaled)  # prepend a column of ones (intercept) required by statsmodels' variance_inflation_factor
    global_vifs = [variance_inflation_factor(X_global.values, i) for i in range(X_global.shape[1])]  # compute the VIF for every column by regressing each against all others
    return global_vifs, X_global  # return both the VIF list and the augmented design matrix (X_global is also used downstream for the condition number)


def build_receptor_vif_table(
    available: list,
    df_scaled: pd.DataFrame,
    global_vifs: list,
) -> pd.DataFrame:
    """
    Build a per-receptor VIF table with global, within-system, and cross-system VIFs.

    Parameters
    ----------
    available   : list of str, features in the analysis.
    df_scaled   : scaled DataFrame.
    global_vifs : list of floats from compute_global_vifs() (index 0 = intercept).

    Returns
    -------
    receptor_df : DataFrame with columns
                  [Receptor, System, Global_VIF, Within_System_VIF, Cross_System_VIF].
    """
    records = []
    for r in available:  # iterate over each receptor/feature in the analysis
        sys_r = SYSTEM_MAPPING.get(r, "Other")  # look up which neurotransmitter system this receptor belongs to; default to 'Other' if not in the mapping (e.g. T1w:T2w)
        same_sys  = [x for x in available if x != r and SYSTEM_MAPPING.get(x, "Other") == sys_r]   # features from the same neurotransmitter system (excluding the target itself)
        other_sys = [x for x in available if x != r and SYSTEM_MAPPING.get(x, "Other") != sys_r]   # features from all other neurotransmitter systems

        records.append({
            "Receptor":          r,
            "System":            sys_r,
            "Global_VIF":        global_vifs[available.index(r) + 1],          # global VIF: index offset by 1 because index 0 is the intercept
            "Within_System_VIF": compute_vif_for_predictor(r, same_sys,  df_scaled),  # VIF when regressed only on same-system receptors; isolates within-system collinearity
            "Cross_System_VIF":  compute_vif_for_predictor(r, other_sys, df_scaled),  # VIF when regressed only on cross-system receptors; isolates between-system collinearity
        })

    return pd.DataFrame(records)  # assemble the per-receptor rows into a tidy DataFrame


def build_correlation_matrix(df_scaled: pd.DataFrame, available: list) -> pd.DataFrame:
    """
    Compute a Pearson correlation matrix, sorted by neurotransmitter system then receptor name.

    Parameters
    ----------
    df_scaled : DataFrame of z-scored features.
    available : list of str, feature column names.

    Returns
    -------
    corr_matrix : DataFrame, shape (n_features, n_features).
    """
    sorted_receptors = sorted(available, key=lambda r: (SYSTEM_MAPPING.get(r, "Other"), r))  # sort receptors first by system (grouping same-system receptors together in the heatmap) then alphabetically within each system
    return df_scaled[sorted_receptors].corr()  # compute the pairwise Pearson correlation matrix on the sorted feature set


def save_csv_outputs(
    csv_dir: str,
    filenames: dict,
    system_df: pd.DataFrame,
    receptor_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
) -> None:
    """Write the three CSV outputs (system metrics, receptor VIFs, correlation matrix) to disk."""
    os.makedirs(csv_dir, exist_ok=True)                                                   # create the output subdirectory if it does not already exist
    system_df.to_csv(  os.path.join(csv_dir, filenames["system"]),      index=False)      # save the global condition number summary (no row index needed)
    receptor_df.to_csv(os.path.join(csv_dir, filenames["receptor"]),    index=False)      # save the per-receptor VIF table (no row index; receptor names are a column)
    corr_matrix.to_csv(os.path.join(csv_dir, filenames["corr_matrix"]))                  # save the correlation matrix with the receptor names as both row and column index


def save_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    fig_dir: str,
    filenames: dict,
) -> None:
    """
    Render and save a Pearson correlation heatmap sorted by neurotransmitter system.

    Parameters
    ----------
    corr_matrix : Pearson correlation matrix DataFrame.
    fig_dir     : Directory where the heatmap PNG will be saved.
    filenames   : Dict as returned by build_filenames().
    """
    os.makedirs(fig_dir, exist_ok=True)  # create the figure output directory if needed

    plt.figure(figsize=(18, 15))  # create a large figure so receptor labels and annotation values don't overlap
    sns.heatmap(
        corr_matrix,          # the correlation matrix to visualise (receptors × receptors)
        annot=True,           # print the Pearson r value inside each cell
        cmap="RdBu_r",        # diverging blue–white–red colormap: blue = negative, red = positive correlation
        center=0,             # anchor the colour scale at r=0 so uncorrelated pairs appear white
        fmt=".2f",            # format annotation values to 2 decimal places
        annot_kws={"size": 8}, # use a small font so numbers fit inside each cell at this figure size
    )
    plt.title("Individual Receptor Spatial Correlations (Group-Clustered)")  # title reflecting that receptors are grouped by neurotransmitter system
    plt.tight_layout()   # adjust layout to prevent axis labels being clipped
    plt.savefig(os.path.join(fig_dir, filenames["heatmap"]))  # write the heatmap PNG to the figure directory
    plt.close()          # release the figure from memory to avoid accumulating open figures


def calculate_multicollinearity(
    X_path: str,
    output_directory_path: str,
    figure_directory_path: str,
    selected_receptors: list = None,
) -> None:
    """
    Multicollinearity analysis for receptor densities and/or T1w/T2w myelination data.

    Assesses collinearity at three levels: globally, within neurotransmitter systems,
    and cross-system — using Variance Inflation Factors. Also computes a global
    condition number and generates a Pearson correlation heatmap sorted by system.

    Parameters
    ----------
    X_path : str
        Path to a CSV file containing receptor and/or myelination data across ROIs.
        Expects an optional 'ROI' index column and one column per feature.
    output_directory_path : str
        Parent directory under which a named subdirectory will be created to store
        all CSV output files. The subdirectory is named after the receptor selection:
        'AllReceptors', 'AllReceptors_T1wT2w', or the receptor names joined by '_'.
    figure_directory_path : str
        Parent directory under which a named subdirectory (same naming logic as
        output_directory_path) will be created to store the correlation heatmap PNG.
    selected_receptors : list of str, optional
        Subset of receptor/feature names to include. If None, all columns are used.
    """
    # 1. Load, filter, and normalise feature data
    df_scaled, available = load_and_select_features(X_path, selected_receptors)  # read the CSV, keep only the requested receptors, and z-score each column

    # 2. Derive output naming — suffix used directly as subdirectory name for both outputs
    suffix    = build_output_suffix(available)   # determine a descriptive suffix (e.g. 'AllReceptors') based on which features are included
    filenames = build_filenames(suffix)          # construct the four output filenames (system CSV, receptor CSV, correlation matrix CSV, heatmap PNG)
    csv_dir   = os.path.join(output_directory_path, suffix)   # full path to the CSV output subdirectory
    fig_dir   = os.path.join(figure_directory_path, suffix)   # full path to the figure output subdirectory

    # 3. Compute global VIFs and condition number
    global_vifs, X_global = compute_global_vifs(df_scaled, available)  # compute a VIF for every column simultaneously and return the augmented design matrix
    system_df = pd.DataFrame({
        "Metric": ["Global_Condition_Number"],  # label for the overall collinearity diagnostic
        "Value":  [np.linalg.cond(X_global)],  # the condition number of the full design matrix; large values (>30) indicate severe collinearity
    })

    # 4. Compute per-receptor VIF table
    receptor_df = build_receptor_vif_table(available, df_scaled, global_vifs)  # for each receptor, compute global, within-system, and cross-system VIFs

    # 5. Compute correlation matrix
    corr_matrix = build_correlation_matrix(df_scaled, available)  # Pearson correlation matrix with receptors sorted by neurotransmitter system for visual grouping

    # 6. Save all outputs
    save_csv_outputs(csv_dir, filenames, system_df, receptor_df, corr_matrix)  # write the three CSV files to the named subdirectory
    save_correlation_heatmap(corr_matrix, fig_dir, filenames)                  # render and save the annotated heatmap PNG

    print(f"CSV outputs saved to:  {csv_dir}")   # log the CSV output location
    print(f"Heatmap saved to:      {fig_dir}")   # log the figure output location