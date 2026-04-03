"""
PLSDA_Analysis.py
-----------------
PLS-DA analysis across binary working memory thresholds with spin and random
permutation testing, plus heatmap visualizations of receptor and system rankings.

Public API
----------
- perform_plsda_across_thresholds()
- visualize_plsda_rank_heatmaps()
- plsda_analysis()
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cross_decomposition import PLSRegression
from neuromaps.nulls import alexander_bloch

from Scripts.utils import (
    SYSTEM_MAPPING,
    ALL_RECEPTOR_COLUMNS,
    T1W_T2W_COLUMN,
    load_schaefer_parcellation_for_spins,
    safe_corr,
)


def load_and_align_data(
    X_path: str,
    Y_path: str,
    selected_receptors: list = None,
    threshold_columns: list = None,
) -> tuple:
    """
    Load X and Y CSVs, optionally filter columns, and align on shared ROIs.

    Parameters
    ----------
    X_path             : path to predictor CSV with 'ROI' index column.
    Y_path             : path to response CSV with 'ROI' index column.
    selected_receptors : predictor columns to keep; None = all columns.
    threshold_columns  : Y columns to analyse; None = all except WorkingMemory_Score.

    Returns
    -------
    X_df              : DataFrame of z-scored predictors, aligned ROIs × features.
    Y_raw             : DataFrame of raw Y columns, aligned ROIs × thresholds.
    threshold_columns : resolved list of threshold column names.
    """
    X_raw = pd.read_csv(X_path).set_index("ROI")  # load predictor CSV and use 'ROI' as the row index
    Y_raw = pd.read_csv(Y_path).set_index("ROI")  # load response CSV and use 'ROI' as the row index

    if selected_receptors is not None:
        X_raw = X_raw[selected_receptors]  # restrict predictors to the requested subset

    excluded = {"WorkingMemory_Score"}  # the continuous WM score is not a binary label and must not be used as a PLS-DA target
    threshold_columns = threshold_columns or [c for c in Y_raw.columns if c not in excluded]  # if no specific columns were requested, use all binary threshold columns, excluding the continuous score

    common_rois = X_raw.index.intersection(Y_raw.index)  # find ROIs that appear in both the predictor and response files to ensure a fully aligned dataset
    X_df  = X_raw.loc[common_rois]   # restrict predictors to the shared ROI set
    Y_raw = Y_raw.loc[common_rois]   # restrict responses to the same shared ROI set

    X_df = pd.DataFrame(
        zscore(X_df, axis=0, nan_policy="omit"),  # z-score each predictor column independently (axis=0 = column-wise); omit NaNs rather than propagating them
        index=X_df.index,    # preserve the ROI row labels
        columns=X_df.columns,  # preserve the feature column names
        dtype=float,           # ensure the result is stored as float64
    )

    return X_df, Y_raw, threshold_columns  # return aligned, z-scored predictors; raw (unscaled) binary responses; and the resolved list of Y columns


def _resolve_run_name(selected_receptors: list = None) -> str:
    """
    Derive a subdirectory name based on which receptor columns are selected.

    Rules
    -----
    - None or all receptors only  → 'AllReceptors'
    - All receptors + T1w:T2w     → 'AllReceptors_T1wT2w'
    - Any other subset            → receptor names joined by underscores

    Parameters
    ----------
    selected_receptors : list of str or None

    Returns
    -------
    name : str
    """
    if selected_receptors is None:
        return "AllReceptors"  # no restriction means all receptors were used

    cols_set     = set(selected_receptors)     # convert to set for fast comparisons
    receptor_set = set(ALL_RECEPTOR_COLUMNS)   # the canonical set of all 19 receptor names

    if cols_set == receptor_set:
        return "AllReceptors"                  # exactly all 19 receptors, no myelination
    elif cols_set == receptor_set | {T1W_T2W_COLUMN}:
        return "AllReceptors_T1wT2w"           # all 19 receptors plus the T1w/T2w myelination proxy
    else:
        return "_".join(selected_receptors)    # a custom subset: join names with underscores for a readable directory name


def resolve_run_directory(output_directory_path: str, selected_receptors: list = None) -> str:
    """
    Create and return the named output subdirectory for this receptor selection.

    Parameters
    ----------
    output_directory_path : parent directory under which the subdirectory is created.
    selected_receptors    : receptor columns selected for this run; used to derive the name.

    Returns
    -------
    run_dir : str, full path to the created subdirectory.
    """
    name    = _resolve_run_name(selected_receptors)  # get the descriptive subdirectory name for this feature selection
    run_dir = os.path.join(output_directory_path, name)  # build the full path by appending the name to the parent directory
    os.makedirs(run_dir, exist_ok=True)  # create the directory; no error if it already exists
    return run_dir  # return the path so the caller can write outputs there


def fit_plsda(
    X: np.ndarray,
    Y: np.ndarray,
    X_columns: pd.Index,
    y_col: str,
) -> tuple:
    """
    Fit a single-component PLS-DA model and compute correlation-based loadings.

    The sign of latent variable 1 (LV1) is flipped if the Y loading is negative,
    so that a positive loading always means more association with the active class.

    Parameters
    ----------
    X         : array, shape (n_rois, n_features), z-scored predictors.
    Y         : array, shape (n_rois, 1), binary 0/1 response.
    X_columns : feature names for the X loading DataFrame.
    y_col     : threshold column name for the Y loading DataFrame.

    Returns
    -------
    xs            : array (n_rois,), X scores for LV1.
    ys            : array (n_rois,), Y scores for LV1.
    loading_x     : array (n_features,), correlation of each X column with xs.
    loading_y     : float, correlation of Y with ys.
    obs_corr      : float, correlation between xs and ys (observed PLS-DA statistic).
    loadings_x_df : DataFrame of X loadings indexed by feature name.
    loadings_y_df : DataFrame of Y loading indexed by y_col.
    """
    pls    = PLSRegression(n_components=1, scale=False).fit(X, Y)  # fit a 1-component PLS regression; scale=False because X is already z-scored; with a binary Y this is equivalent to PLS-DA
    xs, ys = pls.transform(X, Y)  # project X and Y onto the first latent variable to obtain score vectors
    xs, ys = xs[:, 0], ys[:, 0]   # squeeze from (n, 1) to (n,) 1-D arrays for easier downstream operations

    loading_x = np.array([safe_corr(X[:, j], xs) for j in range(X.shape[1])])  # compute correlation-based loadings: Pearson r between each predictor column and the X score vector
    loading_y = safe_corr(Y[:, 0], ys)   # compute the Y loading: Pearson r between the binary response and the Y score vector

    if not np.isnan(loading_y) and loading_y < 0:  # a negative Y loading means LV1 is oriented so that the active class (1) is associated with lower scores; flip sign for interpretability
        xs        *= -1; ys        *= -1   # flip the X and Y score vectors simultaneously so the orientation is consistent
        loading_x *= -1; loading_y *= -1   # flip loadings to match the new score orientation

    obs_corr      = safe_corr(xs, ys)  # the observed PLS-DA statistic: correlation between X and Y scores on LV1 (used as the test statistic in permutation tests)
    loadings_x_df = pd.DataFrame({"LV1": loading_x}, index=X_columns)    # wrap X loadings in a DataFrame with receptor names as the index
    loadings_y_df = pd.DataFrame({"LV1": [loading_y]}, index=[y_col])    # wrap the Y loading in a DataFrame with the threshold column name as the index

    return xs, ys, loading_x, loading_y, obs_corr, loadings_x_df, loadings_y_df  # return scores, loadings, the observed statistic, and the two loading DataFrames


def compute_receptor_ranking(loadings_x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank receptors by their LV1 loading in descending order.

    Parameters
    ----------
    loadings_x_df : DataFrame of X loadings indexed by receptor name.

    Returns
    -------
    receptor_ranking_df : DataFrame with columns [Receptor, Loading_LV1].
    """
    return (
        loadings_x_df[["LV1"]]          # select only the LV1 loading column
        .reset_index()                   # move the receptor-name index to a regular column
        .rename(columns={"index": "Receptor", "LV1": "Loading_LV1"})  # give columns descriptive names
        .sort_values("Loading_LV1", ascending=False)  # sort from highest to lowest loading so the most WM-associated receptor appears first
        .reset_index(drop=True)          # reset the integer row index after sorting
    )


def compute_system_importance(loadings_x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise receptor loadings at the neurotransmitter system level.

    System importance is the mean absolute LV1 loading across all receptors
    belonging to that system.

    Parameters
    ----------
    loadings_x_df : DataFrame of X loadings indexed by receptor name.

    Returns
    -------
    system_summary : DataFrame with columns [System, Mean_Abs_Loading_LV1].
    """
    sys_labels = [SYSTEM_MAPPING.get(r, "Other") for r in loadings_x_df.index]  # look up the neurotransmitter system for each receptor; default to 'Other' for unmapped features like T1w:T2w
    return (
        loadings_x_df.abs()              # take the absolute value of each loading so direction does not affect the system-level summary
        .assign(System=sys_labels)       # add a 'System' column so rows can be grouped by neurotransmitter family
        .groupby("System").mean()        # average absolute loadings within each system
        .reset_index()                   # move the system name from the index to a regular column
        .rename(columns={"LV1": "Mean_Abs_Loading_LV1"})  # rename to a descriptive column header
        .sort_values("Mean_Abs_Loading_LV1", ascending=False)  # sort so the most important system appears first
    )


def run_permutation_tests(
    X: np.ndarray,
    Y: np.ndarray,
    obs_corr: float,
    parc_gii,
    number_permutations: int,
    seed: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Assess PLS-DA significance via spin and random label permutations.

    Spin permutations preserve the spatial autocorrelation of Y by rotating
    the parcellation on the sphere (Alexander-Bloch method). Random permutations
    simply shuffle Y labels, providing a non-spatial null.

    Parameters
    ----------
    X                   : predictor array (n_rois, n_features).
    Y                   : binary response array (n_rois, 1).
    obs_corr            : observed XY score correlation from the true model.
    parc_gii            : Schaefer parcellation GIfTI for spin permutations.
    number_permutations : number of null iterations.
    seed                : random seed passed to alexander_bloch.
    rng                 : numpy Generator for random permutations.

    Returns
    -------
    p_spin   : float, spin-permutation p-value.
    p_random : float, random-permutation p-value.
    """
    Y_spin_nulls = np.asarray(alexander_bloch(
        Y[:, 0], atlas="fsaverage", density="10k",   # the binary WM vector to rotate (1-D)
        parcellation=parc_gii, n_perm=number_permutations, seed=seed,  # Schaefer GIfTI, number of rotations, random seed
    ))  # returns an (n_rois, n_perm) array of spatially rotated null Y vectors; rotations preserve spatial autocorrelation

    null_spin   = np.zeros(number_permutations)  # array to store the XY score correlation for each spin-permuted null model
    null_random = np.zeros(number_permutations)  # array to store the XY score correlation for each randomly permuted null model

    for k in range(number_permutations):  # iterate over each permutation
        for null_Y, null_arr in [
            (Y_spin_nulls[:, k].reshape(-1, 1), null_spin),    # spin null: use the k-th rotated Y vector (preserves spatial autocorrelation)
            (Y[rng.permutation(Y.shape[0])],    null_random),  # random null: independently shuffle the row order of Y (ignores spatial structure)
        ]:
            pls_null    = PLSRegression(n_components=1, scale=False).fit(X, null_Y)  # fit a 1-component PLS model on the null Y
            xs_n, ys_n  = pls_null.transform(X, null_Y)   # project X and null Y onto the first latent variable
            null_arr[k] = safe_corr(xs_n[:, 0], ys_n[:, 0])  # store the XY score correlation as the null statistic for this permutation


    p_spin   = (1 + np.sum(null_spin   >= obs_corr)) / (1 + number_permutations)  # one-sided p-value: proportion of spin nulls at least as extreme as the observed correlation; +1 in numerator and denominator for continuity correction
    p_random = (1 + np.sum(null_random >= obs_corr)) / (1 + number_permutations)  # same formula for the random-permutation null

    return p_spin, p_random  # return both p-values so the caller can report spatial and non-spatial significance separately


def save_threshold_outputs(
    thr_dir: str,
    y_col: str,
    loadings_x_df: pd.DataFrame,
    loadings_y_df: pd.DataFrame,
    receptor_ranking_df: pd.DataFrame,
    system_summary: pd.DataFrame,
    p_df: pd.DataFrame,
) -> None:
    """
    Write all per-threshold CSVs (loadings, rankings, p-values) to thr_dir.

    Parameters
    ----------
    thr_dir             : directory where all files for this threshold are saved.
    y_col               : threshold column name, used to construct filenames.
    loadings_x_df       : DataFrame of X loadings.
    loadings_y_df       : DataFrame of Y loading.
    receptor_ranking_df : receptor ranking DataFrame.
    system_summary      : system importance DataFrame.
    p_df                : p-value DataFrame.
    """
    os.makedirs(thr_dir, exist_ok=True)  # create the per-threshold subdirectory if it does not already exist
    loadings_x_df.to_csv(      os.path.join(thr_dir, f"XLoadings_{y_col}_PLSDA.csv"))                    # per-receptor LV1 loadings (correlation with X scores)
    loadings_y_df.to_csv(      os.path.join(thr_dir, f"YLoadings_{y_col}_PLSDA.csv"))                    # LV1 loading for the binary WM response (correlation with Y scores)
    receptor_ranking_df.to_csv(os.path.join(thr_dir, f"Receptor_Ranking_{y_col}_PLSDA.csv"),  index=False)  # receptors sorted by loading magnitude, descending
    system_summary.to_csv(     os.path.join(thr_dir, f"System_Importance_{y_col}_PLSDA.csv"), index=False)  # neurotransmitter systems sorted by mean absolute loading
    p_df.to_csv(               os.path.join(thr_dir, f"PValues_{y_col}_PLSDA.csv"),           index=False)  # spin and random permutation p-values for LV1


def save_summary_outputs(
    run_dir: str,
    all_summary_rows: list,
    receptor_rankings_across: dict,
    system_rankings_across: dict,
    failed_thresholds: list,
) -> None:
    """
    Write cross-threshold summary CSVs (robustness table, rankings, failures) to run_dir.

    Parameters
    ----------
    run_dir                  : directory where summary files are saved.
    all_summary_rows         : list of per-threshold result dicts.
    receptor_rankings_across : dict {threshold: list of 'Receptor (loading)' strings}.
    system_rankings_across   : dict {threshold: list of 'System (loading)' strings}.
    failed_thresholds        : list of dicts with keys 'Threshold' and 'Error'.
    """
    if all_summary_rows:
        pd.DataFrame(all_summary_rows).sort_values("Threshold").to_csv(
            os.path.join(run_dir, "PLSDA_Threshold_Robustness_Summary.csv"), index=False  # one row per threshold with observed correlation and both p-values; sorted for readability
        )
    if receptor_rankings_across:
        pd.DataFrame({t: pd.Series(v) for t, v in receptor_rankings_across.items()}).to_csv(
            os.path.join(run_dir, "Receptor_Ranking_By_Threshold.csv"), index=False  # wide-format table: one column per threshold, rows = receptor rank positions with loading values embedded as 'Name (loading)' strings
        )
    if system_rankings_across:
        pd.DataFrame({t: pd.Series(v) for t, v in system_rankings_across.items()}).to_csv(
            os.path.join(run_dir, "System_Ranking_By_Threshold.csv"), index=False  # same wide-format structure for neurotransmitter systems
        )
    if failed_thresholds:
        pd.DataFrame(failed_thresholds).to_csv(
            os.path.join(run_dir, "PLSDA_Threshold_Failures.csv"), index=False  # log any threshold that raised an exception so failures are traceable
        )


def perform_plsda_across_thresholds(
    X_path: str,
    Y_path: str,
    output_directory_path: str,
    number_permutations: int = 1000,
    seed: int = 1234,
    threshold_columns: list = None,
    selected_receptors: list = None,
) -> None:
    """
    PLS-DA across binary working memory thresholds with spin and random permutation testing.

    For each binary threshold column in Y_path, fits a PLS-DA model, computes
    X/Y loadings, receptor and system importance rankings, and assesses significance
    via spin and random permutation testing. Results are saved per threshold into
    subdirectories, with combined summary tables written to the output root.

    Parameters
    ----------
    X_path                : path to predictor CSV with 'ROI' index column.
    Y_path                : path to response CSV with 'ROI' index column.
    output_directory_path : root directory for all outputs.
    number_permutations   : number of spin and random permutations. Default 1000.
    seed                  : random seed for reproducibility. Default 1234.
    threshold_columns     : Y columns to analyse; None = all except WorkingMemory_Score.
    selected_receptors    : predictor columns to include; None = all columns.
    """
    X_df, Y_raw, threshold_columns = load_and_align_data(
        X_path, Y_path, selected_receptors, threshold_columns  # load, filter, align, and z-score the predictor and response data
    )
    X = X_df.to_numpy(dtype=float)  # convert the predictor DataFrame to a plain NumPy array for passing to sklearn

    run_dir  = resolve_run_directory(output_directory_path, selected_receptors)  # create and return the named output subdirectory for this run

    parc_gii = load_schaefer_parcellation_for_spins()  # load the Schaefer-100 GIfTI parcellation needed by alexander_bloch() for spin permutations
    rng      = np.random.default_rng(seed)             # create a seeded random number generator for reproducible random-label permutations

    all_summary_rows         = []   # will accumulate one dict per successful threshold (for the robustness summary CSV)
    failed_thresholds        = []   # will accumulate error records for any threshold that raises an exception
    receptor_rankings_across = {}   # will map threshold name → ordered list of 'Receptor (loading)' strings
    system_rankings_across   = {}   # will map threshold name → ordered list of 'System (loading)' strings

    for y_col in threshold_columns:  # iterate over each binary WM threshold column
        try:
            Y = Y_raw[y_col].to_numpy(dtype=float).reshape(-1, 1)  # extract the binary response vector and reshape to (n, 1) as required by PLSRegression

            unique_vals = np.unique(Y[~np.isnan(Y)])   # find unique non-NaN values to validate that the column is truly binary
            if not np.all(np.isin(unique_vals, [0, 1])):
                raise ValueError(f"Column '{y_col}' must be binary 0/1. Found: {unique_vals}")  # reject any column that is not strictly binary
            if len(unique_vals) < 2:
                raise ValueError(f"Column '{y_col}' contains only one class.")  # PLS-DA requires both classes to be present

            xs, ys, _, _, obs_corr, loadings_x_df, loadings_y_df = fit_plsda(
                X, Y, X_df.columns, y_col  # fit the PLS-DA model and compute correlation-based loadings
            )

            receptor_ranking_df = compute_receptor_ranking(loadings_x_df)  # rank receptors by LV1 loading, highest first
            system_summary      = compute_system_importance(loadings_x_df)  # summarise importance at the neurotransmitter system level

            p_spin, p_random = run_permutation_tests(
                X, Y, obs_corr, parc_gii, number_permutations, seed, rng  # run spin and random permutation tests and return both p-values
            )

            p_df = pd.DataFrame({
                "Threshold":      [y_col],      # the name of the binary threshold column being tested
                "LV":             ["LV1"],       # the latent variable (always LV1 for this single-component model)
                "Observed_Corr":  [obs_corr],   # the observed XY score correlation (the PLS-DA test statistic)
                "P_Value_Spin":   [p_spin],     # spin-permutation p-value (spatially constrained null)
                "P_Value_Random": [p_random],   # random-permutation p-value (non-spatial null)
                "Class_0_Count":  [int((Y[:, 0] == 0).sum())],  # number of ROIs in the inactive class
                "Class_1_Count":  [int((Y[:, 0] == 1).sum())],  # number of ROIs in the active class
            })

            thr_dir = os.path.join(run_dir, y_col)  # per-threshold subdirectory: one folder per threshold column
            save_threshold_outputs(
                thr_dir, y_col, loadings_x_df, loadings_y_df,  # save X and Y loadings CSVs
                receptor_ranking_df, system_summary, p_df,       # save receptor ranking, system importance, and p-value CSVs
            )

            all_summary_rows.append({
                "Threshold":          y_col,       # threshold column name
                "Observed_Corr_LV1":  obs_corr,   # observed test statistic
                "P_Value_Spin_LV1":   p_spin,     # spin p-value
                "P_Value_Random_LV1": p_random,   # random p-value
                "Class_0_Count":      int((Y[:, 0] == 0).sum()),  # class size for sanity checking
                "Class_1_Count":      int((Y[:, 0] == 1).sum()),  # class size for sanity checking
            })
            receptor_rankings_across[y_col] = [
                f"{row.Receptor} ({row.Loading_LV1:.2f})"   # format each receptor as 'Name (loading)' for the cross-threshold ranking table
                for row in receptor_ranking_df.itertuples()
            ]
            system_rankings_across[y_col] = [
                f"{row.System} ({row.Mean_Abs_Loading_LV1:.2f})"  # format each system as 'System (mean_abs_loading)' for the cross-threshold ranking table
                for row in system_summary.itertuples()
            ]

        except Exception as e:
            print(f"Failed for '{y_col}': {e}")  # print the error so the user can diagnose which threshold failed and why
            failed_thresholds.append({"Threshold": y_col, "Error": str(e)})  # log the failure without stopping the loop so remaining thresholds are still processed

    save_summary_outputs(
        run_dir, all_summary_rows,               # cross-threshold robustness summary
        receptor_rankings_across, system_rankings_across,  # wide-format ranking tables
        failed_thresholds,                        # error log for any failed thresholds
    )

    print(f"All analyses complete. Results saved to: {run_dir}")  # confirm completion and report the output location


def extract_threshold_value(name: str) -> float:
    """
    Convert a threshold column name (e.g. 'Thr_0p001') to a sortable float.

    The pattern 'Thr_<int>p<decimals>' is parsed into '<int>.<decimals>';
    returns NaN for names that don't match, so unrecognised columns sort last.

    Parameters
    ----------
    name : str, threshold column name.

    Returns
    -------
    value : float
    """
    m = re.match(r"Thr_(\d+)(?:p(\d+))?$", str(name))  # match the pattern 'Thr_<integer>' or 'Thr_<integer>p<decimals>'; the 'p' stands for decimal point
    return float(f"{m.group(1)}.{m.group(2) or 0}") if m else np.nan  # reconstruct the float (e.g. '0p001' → 0.001); return NaN for non-matching strings so they sort last


def load_ordered_thresholds(summary_path: str) -> list:
    """
    Load the robustness summary CSV and return threshold names sorted by numeric value.

    Parameters
    ----------
    summary_path : path to PLSDA_Threshold_Robustness_Summary.csv.

    Returns
    -------
    ordered_thresholds : list of str, e.g. ['Thr_0', 'Thr_0p001', 'Thr_0p01', 'Thr_0p1'].
    """
    summary_df = pd.read_csv(summary_path)  # load the cross-threshold summary produced by perform_plsda_across_thresholds
    summary_df["ThresholdValue"] = summary_df["Threshold"].apply(extract_threshold_value)  # add a numeric column by parsing each threshold name into a float for correct numerical sorting
    return summary_df.sort_values("ThresholdValue")["Threshold"].tolist()  # sort by the numeric value and return the original string names in that order


def parse_ranking_df(
    ranking_df: pd.DataFrame,
    ordered_thresholds: list,
) -> tuple:
    """
    Parse 'Entity (loading)' strings from a ranking CSV into rank and loading matrices.

    Each cell in a ranking CSV contains a string like 'D1 (0.83)'. This function
    extracts the entity name and loading value, then assembles two matrices:
    one for rank position (1 = highest loading) and one for raw loading values.

    Parameters
    ----------
    ranking_df         : DataFrame loaded from Receptor_Ranking_By_Threshold.csv or
                         System_Ranking_By_Threshold.csv.
    ordered_thresholds : threshold column names in numeric order.

    Returns
    -------
    rank_mat    : DataFrame (entities × thresholds) of rank positions.
    loading_mat : DataFrame (entities × thresholds) of loading values.
    """
    cols = [c for c in ordered_thresholds if c in ranking_df.columns]  # filter to only thresholds that are actually present in this CSV (guards against missing threshold runs)

    parsed = {
        col: [
            (m.group(1).strip(), float(m.group(2)))   # successfully parsed: extract entity name and numeric loading value
            if (m := re.match(r"(.+?)\s*\(([-+]?\d*\.?\d+)\)", str(v)))  # match 'Name (±float)' pattern using a walrus operator for inline assignment
            else (str(v).strip(), np.nan)              # fallback: treat the entire cell as the entity name with NaN loading
            for v in ranking_df[col].dropna()         # iterate over non-NaN cells in this threshold column
        ]
        for col in cols  # build the parsed dict for every threshold column
    }

    entities = list(dict.fromkeys(e for col in cols for e, _ in parsed[col]))  # collect all unique entity names in their first-appearance order, preserving the ranking order of the first threshold

    rank_mat    = pd.DataFrame(np.nan, index=entities, columns=cols)  # initialise the rank matrix with NaN; will be filled with integer rank positions (1 = best)
    loading_mat = pd.DataFrame(np.nan, index=entities, columns=cols)  # initialise the loading matrix with NaN; will be filled with raw LV1 loading values

    for col in cols:  # fill both matrices column by column (one threshold at a time)

        for rank, (entity, loading) in enumerate(parsed[col], start=1):  # enumerate from 1 so rank 1 = best (highest loading)
            rank_mat.loc[entity, col]    = rank     # store the integer rank position for this entity and threshold
            loading_mat.loc[entity, col] = loading  # store the raw LV1 loading value for this entity and threshold

    return rank_mat, loading_mat  # return both matrices for downstream sorting and heatmap rendering


# ── 3. Row Sorting ─────────────────────────────────────────────────────────────

def sort_by_mean_rank(rank_mat: pd.DataFrame, loading_mat: pd.DataFrame) -> tuple:
    """
    Sort both matrices so the entity with the lowest (best) mean rank appears first.

    Parameters
    ----------
    rank_mat    : DataFrame (entities × thresholds) of rank positions.
    loading_mat : DataFrame (entities × thresholds) of loading values.

    Returns
    -------
    rank_mat    : row-sorted rank matrix.
    loading_mat : row-sorted loading matrix (same order as rank_mat).
    """
    sorted_index = rank_mat.mean(axis=1).sort_values().index  # compute the mean rank across all thresholds for each entity, then sort ascending so the consistently best-ranked entity comes first
    return rank_mat.loc[sorted_index], loading_mat.loc[sorted_index]  # reindex both matrices with the same sorted order so they remain aligned


def render_heatmap(
    data: pd.DataFrame,
    text_data: pd.DataFrame,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    title: str,
    output_path: str,
    text_fmt,
) -> None:
    """
    Render a generic annotated heatmap and save it to disk.

    Cell colour is determined by `data`; cell text is determined by `text_data`
    via the `text_fmt` callable. Text colour flips to white for high-magnitude cells.

    Parameters
    ----------
    data           : DataFrame of values that drive the colour scale.
    text_data      : DataFrame of values used by text_fmt to produce cell labels.
    cmap           : matplotlib colormap name.
    vmin, vmax     : colour scale bounds.
    colorbar_label : label for the colorbar axis.
    title          : figure title.
    output_path    : full path (including filename) where the PNG is saved.
    text_fmt       : callable(data, row_idx, col_idx) -> str for cell annotations.
    """
    arr = data.to_numpy(dtype=float)  # convert the DataFrame to a raw NumPy array for imshow; NaN values will render transparently

    # Figure height scales with the number of rows so labels are never cramped
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(data))))  # scale figure height with the number of entities so row labels never overlap
    im = ax.imshow(arr, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax)  # display the heatmap; 'nearest' avoids blurring between discrete values

    ax.set_xticks(np.arange(data.shape[1]))   # place one tick per threshold column
    ax.set_xticklabels(data.columns, rotation=45, ha="right")  # label x-axis ticks with threshold names, rotated 45° to prevent overlap
    ax.set_yticks(np.arange(data.shape[0]))   # place one tick per entity row
    ax.set_yticklabels(data.index)            # label y-axis ticks with entity names (receptor or system)
    ax.set_title(title)                       # overall figure title
    ax.set_xlabel("Threshold")                # x-axis label (threshold columns)
    ax.set_ylabel("Entity")                   # y-axis label (receptors or systems)
    plt.colorbar(im, ax=ax).set_label(colorbar_label)  # add a colorbar and label it with the relevant scale description

    scale = np.nanmax(np.abs(arr)) or 1  # find the maximum absolute value in the array; used to determine whether to use white or black annotation text
    for i in range(data.shape[0]):       # iterate over rows (entities)
        for j in range(data.shape[1]):   # iterate over columns (thresholds)
            v = text_data.iat[i, j]      # get the value from text_data that will drive the annotation string
            if not np.isnan(v):          # skip NaN cells (no meaningful annotation to show)
                # White text on dark cells, black text on light cells
                color = "white" if abs(v) > 0.5 * scale else "black"  # flip to white text when the cell is dark (value > 50% of the maximum) for legibility
                ax.text(j, i, text_fmt(data, i, j),
                        ha="center", va="center", fontsize=8, color=color)  # draw the formatted annotation string centred in the cell

    plt.tight_layout()  # adjust layout to prevent axis labels being clipped
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # save the heatmap at 300 dpi with tight cropping
    plt.close()  # release the figure from memory


def visualize_plsda_rank_heatmaps(
    input_directory: str,
    output_directory: str,
) -> None:
    """
    Generate receptor and system ranking heatmaps from PLS-DA summary outputs.

    Reads the cross-threshold ranking CSVs produced by perform_plsda_across_thresholds
    and generates two annotated heatmaps:
      - Receptor heatmap: colour and annotation = LV1 loading value.
      - System heatmap:   colour = rank position, annotation = rank (loading).

    Parameters
    ----------
    input_directory  : directory containing PLSDA summary CSVs
                       (PLSDA_Threshold_Robustness_Summary.csv,
                        Receptor_Ranking_By_Threshold.csv,
                        System_Ranking_By_Threshold.csv).
    output_directory : parent directory; a subdirectory named after the run
                       (e.g. 'AllReceptors') will be created inside it.
    """
    # Mirror the input subdirectory name under the output root for traceability
    run_dir = os.path.join(output_directory, os.path.basename(input_directory))  # replicate the run subdirectory name under the figure output root so CSV and figure outputs share the same naming convention
    os.makedirs(run_dir, exist_ok=True)  # create the figure output subdirectory if needed

    # 1. Determine the correct numeric ordering of threshold columns
    ordered_thresholds = load_ordered_thresholds(
        os.path.join(input_directory, "PLSDA_Threshold_Robustness_Summary.csv")  # parse threshold names from the summary CSV and sort them numerically (e.g. 0 → 0.001 → 0.01 → 0.1)
    )

    # 2. Parse receptor and system ranking CSVs into rank + loading matrices
    receptor_rank, receptor_loading = parse_ranking_df(
        pd.read_csv(os.path.join(input_directory, "Receptor_Ranking_By_Threshold.csv")),  # read the receptor ranking CSV
        ordered_thresholds,  # use the numerically sorted threshold order for consistent column arrangement
    )
    system_rank, system_loading = parse_ranking_df(
        pd.read_csv(os.path.join(input_directory, "System_Ranking_By_Threshold.csv")),  # read the system ranking CSV
        ordered_thresholds,  # same numerically sorted threshold order
    )

    # 3. Sort rows so the highest-ranked (lowest mean rank) entity appears first
    receptor_rank, receptor_loading = sort_by_mean_rank(receptor_rank, receptor_loading)  # sort receptors so the most consistently high-ranked one is at the top of the heatmap
    system_rank,   system_loading   = sort_by_mean_rank(system_rank,   system_loading)    # same sorting for systems

    # 4. Receptor heatmap — diverging colormap centred at 0, annotated with loading value
    max_abs_load = np.nanmax(np.abs(receptor_loading.to_numpy(dtype=float))) or 1  # find the largest absolute loading to set a symmetric colour scale centred at 0
    render_heatmap(
        data=receptor_loading, text_data=receptor_loading,   # colour and annotate cells with the raw LV1 loading value
        cmap="coolwarm", vmin=-max_abs_load, vmax=max_abs_load,  # symmetric diverging colormap: blue = negative (WM-negative), red = positive (WM-positive) loading
        colorbar_label="Loading value",
        title="Receptor Ranking Across Thresholds",
        output_path=os.path.join(run_dir, "Receptor_Ranking_Heatmap.png"),
        text_fmt=lambda d, i, j: f"{d.iat[i, j]:.2f}",  # annotate each cell with the loading value rounded to 2 decimal places
    )

    # 5. System heatmap — sequential colormap over rank position, annotated with rank (loading)
    max_rank = np.nanmax(system_rank.to_numpy(dtype=float)) or 1  # find the worst (highest) rank to set the upper colour bound
    render_heatmap(
        data=system_rank, text_data=system_rank,   # colour cells by rank position (lower = better = darker in viridis_r)
        cmap="viridis_r", vmin=1, vmax=max_rank,   # reversed viridis so rank 1 (best) is the darkest colour
        colorbar_label="Rank position",
        title="System Ranking Across Thresholds",
        output_path=os.path.join(run_dir, "System_Ranking_Heatmap.png"),
        text_fmt=lambda d, i, j: (
            f"{int(d.iat[i, j])} ({system_loading.iat[i, j]:.2f})"  # annotate with 'rank (loading)' when the loading is available
            if not np.isnan(system_loading.iat[i, j])
            else f"{int(d.iat[i, j])}"  # fallback: show only the rank if the loading is NaN
        ),
    )

    print(f"Saved heatmaps to: {run_dir}")  # confirm output location


def plsda_analysis(
    X_path: str,
    Y_path: str,
    output_directory_path: str,
    figure_directory_path: str,
    number_permutations: int = 1000,
    seed: int = 1234,
    threshold_columns: list = None,
    selected_receptors: list = None,
) -> None:
    """
    Run the full PLS-DA pipeline: analysis across thresholds followed by
    receptor and system ranking heatmap visualization.

    Parameters
    ----------
    X_path                : path to predictor CSV with 'ROI' index column.
    Y_path                : path to response CSV with 'ROI' index column.
    output_directory_path : root directory for all CSV outputs.
    figure_directory_path : root directory for all figure outputs. A subdirectory
                            named after the receptor selection (e.g. 'AllReceptors')
                            will be created inside it.
    number_permutations   : number of spin and random permutations. Default 1000.
    seed                  : random seed for reproducibility. Default 1234.
    threshold_columns     : Y columns to analyse; None = all except WorkingMemory_Score.
    selected_receptors    : predictor columns to include; None = all columns.
    """
    # 1. Run PLS-DA analysis and save CSVs
    perform_plsda_across_thresholds(
        X_path=X_path,
        Y_path=Y_path,
        output_directory_path=output_directory_path,   # root for per-threshold and summary CSV outputs
        number_permutations=number_permutations,        # number of spin and random permutations
        seed=seed,                                      # random seed for reproducibility
        threshold_columns=threshold_columns,            # which Y columns to analyse (None = all binary thresholds)
        selected_receptors=selected_receptors,          # which predictor columns to include (None = all)
    )

    # 2. Resolve the subdirectory name that perform_plsda_across_thresholds created
    #    so visualize_plsda_rank_heatmaps can find the summary CSVs
    run_name        = _resolve_run_name(selected_receptors)  # derive the same subdirectory name used in step 1
    input_directory = os.path.join(output_directory_path, run_name)  # full path to the CSV outputs just written

    # 3. Generate ranking heatmaps from the CSVs just written
    visualize_plsda_rank_heatmaps(
        input_directory=input_directory,        # source directory containing the ranking CSVs
        output_directory=figure_directory_path, # root directory where the heatmap PNGs will be written
    )