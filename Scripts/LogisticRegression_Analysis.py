"""
LogisticRegression_Analysis.py
-------------------------------
Forward feature selection and spin-permutation model selection for ridge
logistic regression, predicting binary working memory parcellation labels.

Public API
----------
- run_forward_selection_with_baseline()
- run_spin_absolute_significance()
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from neuromaps.nulls import alexander_bloch
from neuromaps.images import annot_to_gifti, relabel_gifti
from netneurotools.datasets import fetch_schaefer2018

from Scripts.utils import plot_grid

def _aic(y: np.ndarray, probs: np.ndarray, n_features: int) -> float:
    """Compute AIC from binary log-likelihood."""
    p_clip = np.clip(probs, 1e-15, 1 - 1e-15)  # clip predicted probabilities away from 0 and 1 to prevent log(0) in the likelihood calculation
    ll = np.sum(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))  # compute the binary cross-entropy log-likelihood: sum over all ROIs of y*log(p) + (1-y)*log(1-p)
    return 2 * (n_features + 1) - 2 * ll  # AIC = 2k - 2LL, where k = number of features + 1 (for the intercept); lower AIC indicates a better balance of fit and parsimony


def fit_model(
    df: pd.DataFrame,
    y: pd.Series,
    feats: list,
    cv: int,
) -> tuple:
    """
    Standardize features, fit a ridge logistic regression with CV-selected C,
    and return the in-sample AUC together with the fitted objects.

    Parameters
    ----------
    df   : DataFrame containing at least the columns listed in feats.
    y    : Binary target series aligned with df.
    feats: List of feature column names to use.
    cv   : Number of cross-validation folds for selecting C.

    Returns
    -------
    auc    : In-sample ROC-AUC.
    probs  : Predicted class-1 probabilities.
    model  : Fitted LogisticRegressionCV object.
    scaler : Fitted StandardScaler object.
    best_C : The regularization strength chosen by cross-validation.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats])  # fit the scaler on the current feature subset and z-score each column; scaler is retained so the same transformation can be applied to null data later

    C_GRID = np.logspace(-4, 4, 50)  # 50 regularisation strengths evenly spaced on a log scale from 0.0001 to 10000; covers strong to weak L2 regularisation

    model = LogisticRegressionCV(
        Cs=C_GRID,              # candidate regularisation strengths to search over
        cv=cv,                  # number of cross-validation folds for selecting the best C
        l1_ratios=(0,),         # l1_ratio=0 means pure L2 (ridge) regularisation; no L1 sparsity
        scoring="roc_auc",      # select C by maximising cross-validated AUC
        solver="lbfgs",         # limited-memory BFGS optimiser; suitable for small-to-medium binary classification problems
        max_iter=1000,          # maximum iterations for the optimiser per fold; large enough for convergence with ridge
        random_state=42,        # seed for reproducible fold splits
        refit=True,             # refit the final model on all data using the best C after CV
        use_legacy_attributes=False,  # use the new attribute naming convention (C_ instead of C) introduced in recent sklearn
    ).fit(X, y)

    probs = model.predict_proba(X)[:, 1]  # predicted class-1 probabilities from the refitted model on the full training data
    auc   = roc_auc_score(y, probs)       # in-sample ROC-AUC; note this is optimistic because the model was trained on the same data

    return auc, probs, model, scaler, float(model.C_)  # return the AUC, probabilities, fitted model, fitted scaler, and the CV-selected regularisation strength


def compute_step_metrics(
    df: pd.DataFrame,
    y: pd.Series,
    selected: list,
    auc: float,
    probs: np.ndarray,
    model,
    scaler,
) -> dict:
    """
    Compute a single row of per-step diagnostics.

    Returns a dict containing AUC, AIC, ridge C, confusion-matrix statistics,
    and model coefficients.
    """
    aic  = _aic(np.array(y), probs, len(selected))  # compute AIC using the log-likelihood and the number of selected features
    yhat = (np.clip(probs, 1e-15, 1 - 1e-15) >= 0.5).astype(int)  # convert predicted probabilities to binary class labels using a 0.5 threshold
    ya   = np.array(y)  # convert the target Series to a NumPy array for element-wise comparison

    TP = np.sum((yhat == 1) & (ya == 1))  # true positives: predicted active and actually active
    TN = np.sum((yhat == 0) & (ya == 0))  # true negatives: predicted inactive and actually inactive
    FP = np.sum((yhat == 1) & (ya == 0))  # false positives: predicted active but actually inactive
    FN = np.sum((yhat == 0) & (ya == 1))  # false negatives: predicted inactive but actually active

    return {
        "step":           None,             # placeholder; the caller fills this with the current step number
        "added_feature":  selected[-1],     # the most recently added feature (the one added at this forward-selection step)
        "cumulative_auc": round(auc, 4),    # in-sample AUC after adding this feature
        "aic":            round(aic, 4),    # AIC after adding this feature; used alongside AUC to penalise model complexity
        "ridge_C":        round(float(model.C_), 6),  # the CV-selected regularisation strength for this feature set
        "sensitivity":    round(TP / (TP + FN), 4) if (TP + FN) else np.nan,  # true positive rate (recall); NaN if no positive labels exist
        "specificity":    round(TN / (TN + FP), 4) if (TN + FP) else np.nan,  # true negative rate; NaN if no negative labels exist
        "ppv":            round(TP / (TP + FP), 4) if (TP + FP) else np.nan,  # positive predictive value (precision); NaN if nothing was predicted positive
        "npv":            round(TN / (TN + FN), 4) if (TN + FN) else np.nan,  # negative predictive value; NaN if nothing was predicted negative
        **{f"coef_{f}": round(c, 4) for f, c in zip(selected, model.coef_[0])}  # unpack model coefficients as separate columns named 'coef_<feature>'; coef_[0] is the single row of coefficients for binary classification
    }


def forward_selection(
    df: pd.DataFrame,
    y: pd.Series,
    baseline: str,
    predictor_cols: list,
    cv: int,
) -> tuple:
    """
    Run greedy forward feature selection starting from a single baseline feature.

    At each step the candidate that maximises cross-validated AUC is added.

    Parameters
    ----------
    df            : DataFrame containing all predictor and target columns.
    y             : Binary target series aligned with df.
    baseline      : Column name of the mandatory first feature.
    predictor_cols: Full ordered list of candidate feature names (baseline included).
    cv            : Number of CV folds passed to fit_model.

    Returns
    -------
    rows     : List of metric dicts (one per step).
    roc_data : List of (fpr, tpr, label) tuples for every step's ROC curve.
    """
    selected  = [baseline]  # always start with the mandatory baseline feature (e.g. T1w:T2w myelination)
    remaining = [p for p in predictor_cols if p != baseline]  # all other candidate features that have not yet been added

    rows     = []  # will accumulate one diagnostic dict per forward-selection step
    roc_data = []  # will accumulate (fpr, tpr, label) tuples for ROC curve plotting

    auc, probs, model, scaler, best_C = fit_model(df, y, selected, cv)  # fit the baseline model (step 0) with just the mandatory feature
    row = compute_step_metrics(df, y, selected, auc, probs, model, scaler)  # compute diagnostics for the baseline model
    row["step"] = len(rows)  # step 0
    rows.append(row)  # save the baseline step metrics

    fpr, tpr, _ = roc_curve(y, probs)  # compute the ROC curve for the baseline model
    roc_data.append((fpr, tpr, f"{' + '.join(selected)} (AUC={auc:.3f}, AIC={row['aic']:.1f})"))  # store the ROC curve with a descriptive label for plotting

    while remaining:  # continue adding features until no candidates remain
        results = {}
        for feat in remaining:  # try adding each remaining feature individually to find the best candidate
            try:
                results[feat] = fit_model(df, y, selected + [feat], cv)  # fit a model with this candidate added and store its results
            except Exception:
                continue  # skip any feature that causes the model to fail (e.g. singular matrix edge cases)

        if not results:
            break  # if no candidates could be fitted successfully, stop the selection loop

        best_feat = max(results, key=lambda f: results[f][0])  # pick the candidate that produced the highest AUC (greedy criterion)
        auc, probs, model, scaler, best_C = results[best_feat]  # retrieve the fitted model and metrics for the best candidate
        selected.append(best_feat)    # add the best candidate to the selected feature set
        remaining.remove(best_feat)   # remove it from the pool of remaining candidates

        print(f"  Added {best_feat} (AUC={auc:.4f}, C={best_C:.4f})")  # log which feature was added and the resulting AUC and regularisation strength

        row = compute_step_metrics(df, y, selected, auc, probs, model, scaler)  # compute diagnostics for the updated model
        row["step"] = len(rows)  # record the current step number
        rows.append(row)         # save the step metrics

        fpr, tpr, _ = roc_curve(y, probs)  # compute the ROC curve for this step's model
        roc_data.append((fpr, tpr, f"{' + '.join(selected)} (AUC={auc:.3f}, AIC={row['aic']:.1f})"))  # store the ROC curve with a descriptive label

    return rows, roc_data  # return all step metrics and ROC curve data for downstream saving and plotting


def plot_forward_selection_roc(
    roc_data: list,
    target: str,
    baseline: str,
    cv: int,
    save_path: str,
) -> None:
    """
    Draw one ROC curve per forward-selection step on a single axes and save to disk.

    Parameters
    ----------
    roc_data  : List of (fpr, tpr, label) tuples produced by forward_selection().
    target    : Name of the binary outcome variable (used in the title).
    baseline  : Baseline feature name (used in the title).
    cv        : Number of CV folds (used in the title).
    save_path : Full file path where the PNG will be written.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")  # dashed diagonal line representing a random classifier (AUC = 0.5)

    for fpr, tpr, label in roc_data:
        ax.plot(fpr, tpr, label=label)  # plot one ROC curve per forward-selection step; each curve adds one more feature

    ax.set_xlabel("False Positive Rate")   # x-axis: 1 - specificity
    ax.set_ylabel("True Positive Rate")    # y-axis: sensitivity
    ax.set_title(f"Ridge Forward Selection - {target}")  # title identifies the binary WM threshold being classified
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # place the legend outside the axes to avoid overlapping the curves
    fig.tight_layout()   # adjust layout before saving to prevent the external legend from being clipped
    fig.savefig(save_path, dpi=300, bbox_inches="tight")  # save at 300 dpi; bbox_inches='tight' includes the external legend
    plt.close(fig)  # release the figure from memory


def plot_combined_roc(
    final_roc_per_target: dict,
    baseline: str,
    cv: int,
    save_path: str,
) -> None:
    """
    Draw the final-step ROC curve for every target on one figure and save to disk.

    Parameters
    ----------
    final_roc_per_target : Dict {target_name: (fpr, tpr, auc, aic)}.
    baseline             : Baseline feature name (title).
    cv                   : Number of CV folds (title).
    save_path            : Full file path for the PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")  # random classifier baseline

    for target, (fpr, tpr, auc, aic) in final_roc_per_target.items():
        ax.plot(fpr, tpr, lw=2, label=f"{target} (AUC={auc:.3f}, AIC={aic:.1f})")  # plot the final model ROC curve for each WM threshold; line width=2 for visibility

    ax.set_xlabel("False Positive Rate")   # x-axis: 1 - specificity
    ax.set_ylabel("True Positive Rate")    # y-axis: sensitivity
    ax.set_title("Ridge Forward Selection — Final Model per Threshold")  # title clarifies this shows only the final (full) model per threshold
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # external legend to avoid overlapping curves
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_steps_csv(rows: list, target: str, run_dir: str) -> None:
    """
    Convert per-step metric dicts to a tidy DataFrame and write to CSV.

    Parameters
    ----------
    rows    : List of dicts produced by forward_selection().
    target  : Target name used to construct the file name.
    run_dir : Directory where the CSV will be saved.
    """
    steps_df  = pd.DataFrame(rows)  # convert the list of per-step dicts to a DataFrame (one row per forward-selection step)
    meta_cols = ["step", "added_feature", "cumulative_auc", "aic", "ridge_C",
                 "sensitivity", "specificity", "ppv", "npv"]  # fixed metadata columns that appear in every step
    coef_cols = sorted(c for c in steps_df.columns if c.startswith("coef_"))  # dynamically find all coefficient columns (one per selected feature); sort for a consistent column order
    steps_df.reindex(columns=meta_cols + coef_cols).to_csv(
        os.path.join(run_dir, f"{target}_steps.csv"), index=False  # reorder columns (meta first, then coefficients) and save to CSV without the default integer row index
    )


def run_forward_selection_with_baseline(
    X_path: str,
    y_path: str,
    save_dir: str,
    figures_dir: str,
    predictor_cols: list,
    targets: list,
    baseline: str = "T1w:T2w",
    cv: int = 5,
) -> None:
    """
    Load data, loop over targets, run greedy forward selection and generate
    all figures and CSV outputs.

    Parameters
    ----------
    X_path        : Path to the CSV of predictor features.
    y_path        : Path to the CSV of binary target columns.
    save_dir      : Root directory for CSV outputs.
    figures_dir   : Root directory for figure outputs.
    predictor_cols: Ordered list of feature column names to include.
    targets       : List of binary outcome column names in y_path.
    baseline      : Feature always included first (default: "T1w:T2w").
    cv            : Number of CV folds for LogisticRegressionCV (default: 5).
    """
    X_raw = pd.read_csv(X_path)[predictor_cols].apply(pd.to_numeric, errors="coerce")  # load predictors and coerce to numeric, replacing any non-numeric entries with NaN
    Y_raw = pd.read_csv(y_path)[targets].apply(pd.to_numeric, errors="coerce")          # load binary target columns and coerce to numeric

    subdir_name = "_".join(predictor_cols)  # create a unique subdirectory name by joining all predictor names so different feature sets are stored separately
    run_dir  = os.path.join(save_dir,    subdir_name)  # full path to the CSV output subdirectory for this feature set
    fig_root = os.path.join(figures_dir, subdir_name)  # full path to the figure output subdirectory for this feature set
    os.makedirs(run_dir,  exist_ok=True)  # create the CSV subdirectory if needed
    os.makedirs(fig_root, exist_ok=True)  # create the figure subdirectory if needed

    final_roc_per_target = {}  # will store the final-step ROC curve data for each target, used to build the combined ROC plot

    for target in targets:  # iterate over each binary WM threshold to run a separate forward selection
        print(f"\n── {target} ──")  # log the current target for progress tracking
        df = pd.concat([X_raw, Y_raw[target]], axis=1).dropna()  # combine predictors with this target column and drop any ROI rows with NaN in any column
        y  = df[target].astype(int)  # extract the binary target as an integer array (0 / 1)

        rows, roc_data = forward_selection(df, y, baseline, predictor_cols, cv)  # run greedy forward selection starting from the baseline feature

        save_steps_csv(rows, target, run_dir)  # write per-step metrics (AUC, AIC, coefficients, etc.) to a CSV

        per_target_png = os.path.join(fig_root, f"{target}_forward_selection.png")  # path for this target's per-step ROC figure
        plot_forward_selection_roc(roc_data, target, baseline, cv, per_target_png)  # draw one ROC curve per step and save the figure

        fpr_final, tpr_final, _ = roc_data[-1]  # extract the final-step ROC curve (last entry corresponds to the full model)
        final_auc = rows[-1]["cumulative_auc"]   # AUC of the final model (all features selected)
        final_aic = rows[-1]["aic"]              # AIC of the final model
        final_roc_per_target[target] = (fpr_final, tpr_final, final_auc, final_aic)  # store for the combined ROC plot

    combined_roc_path = os.path.join(fig_root, "all_thresholds_final_model_ROC.png")  # path for the combined figure showing final models for all thresholds
    plot_combined_roc(final_roc_per_target, baseline, cv, combined_roc_path)           # draw all final-model ROC curves on one figure

    grid_path = os.path.join(fig_root, "all_thresholds_forward_selection_grid.png")  # path for the summary grid figure
    plot_grid(targets, fig_root, grid_path, filename_fmt="{target}_forward_selection.png")  # assemble all per-target per-step ROC figures into a 2-column grid

    print(f"Results saved to: {run_dir}")    # log the CSV output location
    print(f"Figures saved to: {fig_root}")  # log the figure output location


def fit_empirical_models(
    df: pd.DataFrame,
    y: np.ndarray,
    named_models: dict,
    cv: int,
    seed: int,
) -> dict:
    """
    Standardise features and fit one ridge logistic regression per model using
    cross-validation to select the best regularisation strength C.

    Parameters
    ----------
    df           : DataFrame containing all predictor columns.
    y            : Binary target array aligned with df.
    named_models : Dict {model_name: [feature_col, ...]} defining each model.
    cv           : Number of CV folds for selecting C.
    seed         : Random seed for reproducibility.

    Returns
    -------
    empirical_metrics : Dict {model_name: {feats, scaler, best_c, emp_auc,
                        aic, fpr, tpr}}.
    """
    empirical_metrics = {}  # will accumulate fitted model data for each named model
    C_GRID = np.logspace(-4, 4, 50)  # 50 regularisation strengths on a log scale; same grid as in fit_model() for consistency

    for m_name, feats in named_models.items():  # iterate over each predefined model configuration
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(df[feats].values)  # fit and apply z-score normalisation to this model's feature subset

        model_cv = LogisticRegressionCV(
            Cs=C_GRID,           # candidate regularisation strengths to search
            cv=cv,               # number of CV folds for selecting C
            l1_ratios=(0,),      # pure L2 (ridge) regularisation
            scoring="roc_auc",   # select C by maximising cross-validated AUC
            solver="lbfgs",      # BFGS optimiser
            max_iter=1000,       # maximum iterations per fold
            random_state=seed,   # seeded for reproducibility
            use_legacy_attributes=False,  # use updated sklearn attribute naming
        ).fit(X_sc, y)

        best_c  = float(model_cv.C_)              # the regularisation strength selected by cross-validation
        probs   = model_cv.predict_proba(X_sc)[:, 1]  # in-sample class-1 predicted probabilities
        emp_auc = roc_auc_score(y, probs)          # in-sample AUC (used as the empirical test statistic)
        fpr, tpr, _ = roc_curve(y, probs)          # ROC curve coordinates for plotting
        aic = _aic(y, probs, len(feats))            # AIC for this model

        empirical_metrics[m_name] = {
            "feats":   feats,    # the feature list for this model (needed to transform null data with the same columns)
            "scaler":  scaler,   # the fitted scaler (needed to apply the same normalisation to null data)
            "best_c":  best_c,   # the CV-selected C (used to fix regularisation when refitting on null data)
            "emp_auc": emp_auc,  # the empirical AUC (the test statistic to compare against the null distribution)
            "aic":     aic,      # the empirical AIC (for reporting and model comparison)
            "fpr":     fpr,      # ROC curve false positive rates (for plotting)
            "tpr":     tpr,      # ROC curve true positive rates (for plotting)
        }
        print(f" {m_name}: AUC = {emp_auc:.4f}, AIC = {aic:.2f} (C = {best_c:.4f})")  # log each model's performance for progress tracking

    return empirical_metrics  # return the dict of fitted model data for use in permutation testing and plotting


def generate_spin_nulls(
    y: np.ndarray,
    parc_gii,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    """
    Generate spatially constrained null distributions via Alexander-Bloch spin test.

    Parameters
    ----------
    y        : Binary target array of shape (n_parcels,).
    parc_gii : Schaefer-100 GIfTI parcellation for the spin test.
    n_perm   : Number of spin permutations to generate.
    seed     : Random seed for reproducibility.

    Returns
    -------
    y_spins : Binary array of shape (n_parcels, n_perm).
    """
    y_spins = alexander_bloch(
        y, atlas="fsaverage", density="10k",   # rotate the continuous y values on the fsaverage sphere at 10k vertex density
        parcellation=parc_gii, n_perm=n_perm, seed=seed,  # Schaefer GIfTI for parcel centres, number of rotations, random seed
    )
    return (y_spins > 0.5).astype(int)  # threshold the rotated continuous values at 0.5 to recover binary class labels (preserving the class boundary)


def run_permutation_loop(
    df: pd.DataFrame,
    y_spins: np.ndarray,
    empirical_metrics: dict,
    n_perm: int,
    seed: int,
) -> dict:
    """
    Refit each model on every spin-permuted target and record the null AUCs.

    Parameters
    ----------
    df               : DataFrame with predictor columns.
    y_spins          : Binary array of shape (n_parcels, n_perm).
    empirical_metrics: Dict returned by fit_empirical_models().
    n_perm           : Number of permutations.
    seed             : Random seed for LogisticRegression.

    Returns
    -------
    null_aucs : Dict {model_name: np.array of shape (n_perm,)} with NaN for
                permutations that produced single-class targets.
    """
    model_names = list(empirical_metrics.keys())  # ordered list of model names for consistent iteration
    null_aucs   = {m: np.full(n_perm, np.nan) for m in model_names}  # initialise null AUC arrays with NaN; permutations that fail (single-class) will remain NaN

    for k in range(n_perm):  # iterate over each spin permutation
        y_null = y_spins[:, k]  # the k-th rotated binary null target vector
        if len(np.unique(y_null)) < 2:
            continue  # skip permutations where rotation produces a single-class target (AUC is undefined)

        for m_name, emp_data in empirical_metrics.items():  # refit every model on this null target
            X_sc = emp_data["scaler"].transform(df[emp_data["feats"]].values)  # apply the same scaler fitted on the empirical data; ensures null models see the same feature scale

            model_null = LogisticRegression(
                C=emp_data["best_c"],  # fix C to the empirically selected value; avoids re-running CV for every permutation (computationally expensive)
                solver="lbfgs",        # same optimiser as the empirical model
                max_iter=1000,         # same iteration limit
                random_state=seed,     # seeded for reproducibility
            ).fit(X_sc, y_null)  # fit logistic regression on the null (spin-rotated) target

            probs_null = model_null.predict_proba(X_sc)[:, 1]  # class-1 predicted probabilities under the null
            null_aucs[m_name][k] = roc_auc_score(y_null, probs_null)  # store the null AUC as one sample in the null distribution for this model

    return null_aucs  # return the null AUC arrays (one per model) for computing p-values


def compute_p_values(empirical_metrics: dict, null_aucs: dict) -> dict:
    """
    Compute one-sided spin-test p-values for each model.

    p = (1 + #{null AUCs >= empirical AUC}) / (1 + n_valid_nulls)

    Returns
    -------
    p_values : Dict {model_name: float}.
    """
    p_values = {}
    for m_name, emp_data in empirical_metrics.items():  # compute a p-value for every model
        valid_nulls = null_aucs[m_name][~np.isnan(null_aucs[m_name])]  # exclude NaN entries (single-class permutations) from the null distribution
        p_values[m_name] = (1 + np.sum(valid_nulls >= emp_data["emp_auc"])) / (1 + len(valid_nulls))  # one-sided p-value with continuity correction: proportion of valid null AUCs at least as large as the empirical AUC

    return p_values  # return the p-value for each model


def collect_results_row(
    m_name: str,
    target: str,
    empirical_metrics: dict,
    p_values: dict,
) -> dict:
    """
    Build one results row dict for a given model / target combination.

    Returns
    -------
    dict with columns: Internal_Model_Name, Target, Features,
                       Empirical_AUC, AIC, P_value_AUC.
    """
    emp_data = empirical_metrics[m_name]  # retrieve the fitted model data for this model name
    return {
        "Internal_Model_Name": m_name,                          # internal key used to split results by model when saving CSVs; dropped before user-facing output
        "Target":              target,                           # the binary WM threshold column being classified
        "Features":            " + ".join(emp_data["feats"]),   # human-readable list of features in this model
        "Empirical_AUC":       round(emp_data["emp_auc"], 4),   # in-sample AUC of the empirical model
        "AIC":                 round(emp_data["aic"], 4),       # AIC of the empirical model
        "P_value_AUC":         round(p_values[m_name], 4),      # spin-test p-value for the empirical AUC
    }


def plot_target_roc(
    empirical_metrics: dict,
    p_values: dict,
    target: str,
    save_path: str,
) -> None:
    """
    Draw one ROC curve per model on a single axes and save to disk.

    The legend includes AUC, AIC, and the spin-test p-value for each model.

    Parameters
    ----------
    empirical_metrics : Dict returned by fit_empirical_models().
    p_values          : Dict returned by compute_p_values().
    target            : Target name used in the figure title.
    save_path         : Full file path where the PNG will be written.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")  # diagonal reference line for a random classifier

    for m_name, emp_data in empirical_metrics.items():
        short_name = m_name.split("_")[0]  # shorten the model name (e.g. 'M1') for a more compact legend label
        label = (
            f"{short_name} "
            f"(AUC={emp_data['emp_auc']:.3f}, "   # empirical AUC to 3 decimal places
            f"AIC={emp_data['aic']:.1f}, "         # AIC to 1 decimal place
            f"p={p_values[m_name]:.3f})"           # spin-test p-value to 3 decimal places
        )
        ax.plot(emp_data["fpr"], emp_data["tpr"], lw=2, label=label)  # plot this model's ROC curve with a descriptive legend entry

    ax.set_xlabel("False Positive Rate")   # x-axis: 1 - specificity
    ax.set_ylabel("True Positive Rate")    # y-axis: sensitivity
    ax.set_title(f"Spin-Tested Model Comparison — {target}")  # title identifies the WM threshold and the type of significance test
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # external legend to avoid overlapping curves
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_results_csvs(
    results_rows: list,
    model_names: list,
    run_dir: str,
) -> None:
    """
    Split the combined results DataFrame by model and write one CSV per model.

    Parameters
    ----------
    results_rows : List of dicts produced by collect_results_row().
    model_names  : Ordered list of model names (determines file names).
    run_dir      : Directory where the CSVs will be saved.
    """
    results_df = pd.DataFrame(results_rows)  # combine all result dicts into a single DataFrame

    for m_name in model_names:  # write a separate CSV for each model so results are easy to inspect individually
        model_df = results_df[results_df["Internal_Model_Name"] == m_name].drop(
            columns=["Internal_Model_Name"]  # remove the internal key column; it was only needed for filtering and is not meaningful to the end user
        )
        output_csv = os.path.join(run_dir, f"{m_name}.csv")  # name the CSV after the model (e.g. 'M1: D1+DAT.csv')
        model_df.to_csv(output_csv, index=False)  # save without row index
        print(f"  Saved: {output_csv}")  # log each saved file for progress tracking


def run_spin_absolute_significance(
    X_path: str,
    y_path: str,
    save_dir: str,
    figures_dir: str,
    models: list,
    targets: list,
    cv: int = 5,
    n_perm: int = 1000,
    seed: int = 1234,
) -> pd.DataFrame:
    """
    Spin-permutation significance testing for a set of logistic regression models.

    For each target, fits each model empirically, generates spatially constrained
    null distributions via the Alexander-Bloch spin test, and computes one-sided
    p-values for model AUC. All figures and CSV outputs are saved to disk.

    Parameters
    ----------
    X_path      : Path to the CSV of predictor features (rows = ROIs).
    y_path      : Path to the CSV of binary target columns (rows = ROIs).
    save_dir    : Root directory for CSV outputs.
    figures_dir : Root directory for figure outputs.
    models      : List of lists; each inner list defines the features for one model.
    targets     : List of binary outcome column names in y_path.
    cv          : Number of CV folds for LogisticRegressionCV (default 5).
    n_perm      : Number of spin permutations (default 1000).
    seed        : Random seed for spin generation and model fitting (default 1234).

    Returns
    -------
    results_df : DataFrame with one row per model × target combination.
    """
    named_models = {f"M{i+1}: {'+'.join(feats)}": feats for i, feats in enumerate(models)}  # assign descriptive names like 'M1: D1+DAT' to each model for labelling outputs
    model_names  = list(named_models.keys())  # ordered list of model names used for consistent iteration and file naming
    all_cols     = list(dict.fromkeys(c for feats in named_models.values() for c in feats))  # collect all unique predictor columns across all models, in first-appearance order, to load them all in one CSV read

    subdir_name = "_".join("".join(feats) for feats in models)  # build a unique subdirectory name by concatenating all feature sets; distinguishes runs with different model configurations
    run_dir  = os.path.join(save_dir,    subdir_name)  # full path to the CSV output subdirectory
    fig_root = os.path.join(figures_dir, subdir_name)  # full path to the figure output subdirectory
    os.makedirs(run_dir,  exist_ok=True)  # create CSV subdirectory if needed
    os.makedirs(fig_root, exist_ok=True)  # create figure subdirectory if needed

    X_raw = pd.read_csv(X_path).set_index("ROI")[all_cols].apply(pd.to_numeric, errors="coerce")  # load predictors, select only the needed columns, and coerce to numeric (replacing non-numeric with NaN)
    Y_raw = pd.read_csv(y_path).set_index("ROI")[targets].apply(pd.to_numeric, errors="coerce")   # load binary target columns and coerce to numeric

    schaefer_annot = fetch_schaefer2018(version="fsaverage5")["100Parcels7Networks"]  # download the Schaefer-100 fsaverage5 annotation for the spin test
    parc_gii       = relabel_gifti(annot_to_gifti(schaefer_annot))                   # convert to GIfTI and relabel parcel indices; required by alexander_bloch()

    results_rows = []  # will accumulate one result dict per model × target combination

    for target in targets:  # iterate over each binary WM threshold column
        print(f"\n── {target} ──")  # log the current target for progress tracking
        df = pd.concat([X_raw, Y_raw[target]], axis=1).dropna()  # combine predictor matrix with this target column, then drop any ROI rows with NaN in any column
        y  = df[target].values.astype(int)  # extract the binary target as an integer NumPy array

        empirical_metrics = fit_empirical_models(df, y, named_models, cv, seed)  # fit each model on the true data and collect AUC, AIC, ROC curves, scalers, and best C values
        y_spins           = generate_spin_nulls(y, parc_gii, n_perm, seed)        # generate n_perm spatially constrained null binary target vectors via sphere rotation

        null_aucs = run_permutation_loop(df, y_spins, empirical_metrics, n_perm, seed)  # refit each model on every null target and collect the null AUC distribution
        p_values  = compute_p_values(empirical_metrics, null_aucs)                      # compute one-sided spin-test p-values for each model

        for m_name in model_names:
            results_rows.append(collect_results_row(m_name, target, empirical_metrics, p_values))  # build one result row per model and append to the combined results list

        target_png = os.path.join(fig_root, f"{target}_ROC.png")  # path for this target's multi-model ROC figure
        plot_target_roc(empirical_metrics, p_values, target, target_png)  # draw ROC curves for all models with AUC, AIC, and p-value annotations

    grid_path = os.path.join(fig_root, "all_thresholds_ROC_grid.png")  # path for the summary grid figure
    plot_grid(targets, fig_root, grid_path, filename_fmt="{target}_ROC.png")  # assemble all per-target ROC figures into a 2-column grid for easy comparison

    save_results_csvs(results_rows, model_names, run_dir)  # split results by model and save one CSV per model

    print(f"\nFigures saved to:\n  {fig_root}")   # log the figure output location
    print(f"Results saved to:\n  {run_dir}")       # log the CSV output location

    return pd.DataFrame(results_rows).drop(columns=["Internal_Model_Name"])  # return a clean results DataFrame (without the internal model-name key) for optional use by the caller