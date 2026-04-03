"""
utils.py
--------
Shared utilities used across multiple analysis modules.

Contents
--------
- SYSTEM_MAPPING          : receptor → neurotransmitter system constant
- ALL_RECEPTOR_COLUMNS    : canonical list of all receptor names
- T1W_T2W_COLUMN          : T1w/T2w column name constant
- setup_environment()     : add wb_command to PATH and create output directory
- load_schaefer_parcellation()        : fetch Schaefer-100 atlas (ROI names + gifti)
- load_schaefer_surface_parcellation(): fetch Schaefer-100 atlas (roi_map + hemi_labels)
- load_schaefer_parcellation_for_spins(): fetch Schaefer-100 gifti for spin permutations
- map_roi_values_to_vertices()        : project ROI-level values onto vertex arrays
- save_figure()                       : save a matplotlib figure to disk
- safe_corr()                         : NaN-safe Pearson correlation
- plot_grid()                         : assemble per-target PNG files into a grid figure
"""

import os
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuromaps.images import annot_to_gifti, relabel_gifti
from netneurotools.datasets import fetch_schaefer2018


# Maps each receptor name to its neurotransmitter system.
# Used for within- vs cross-system VIF decomposition, system-level importance
# summaries, and output directory naming.
SYSTEM_MAPPING = {
    '5HT1a': 'Serotonin',    '5HT1b': 'Serotonin',  '5HT2a': 'Serotonin',
    '5HT4':  'Serotonin',    '5HT6':  'Serotonin',   '5HTT':  'Serotonin',
    'D1':    'Dopamine',     'D2':    'Dopamine',     'DAT':   'Dopamine',
    'A4B2':  'Acetylcholine', 'M1':   'Acetylcholine', 'VAChT': 'Acetylcholine',
    'mGluR5': 'Glutamate',   'NMDA':  'Glutamate',
    'GABAa': 'GABA',         'H3':    'Histamine',    'NET':   'Noradrenaline',
    'CB1':   'Cannabinoid',  'MOR':   'Opioid',
}

ALL_RECEPTOR_COLUMNS = list(SYSTEM_MAPPING.keys())  # ordered list of all 19 receptor/transporter names; used to distinguish "all-receptor" runs from subsets
T1W_T2W_COLUMN = "T1w:T2w"                          # canonical column name for the myelination proxy derived from T1w/T2w ratio maps


def setup_environment(wb_view_path: str, output_directory_path: str) -> str:
    """
    Add wb_command to PATH and create the output directory.

    Parameters
    ----------
    wb_view_path : str
        Path to the directory containing the wb_command binary.
    output_directory_path : str
        Directory to create if it does not already exist.

    Returns
    -------
    wb : str or None
        Full path to wb_command if found on PATH, otherwise None.
    """
    os.environ["PATH"] = f"{wb_view_path}:" + os.environ.get("PATH", "")  # prepend the wb_command directory to PATH so the binary can be found by subprocess calls
    wb = shutil.which("wb_command")   # verify wb_command is now discoverable on PATH; returns None if still missing
    os.makedirs(output_directory_path, exist_ok=True)  # create the output directory if it does not already exist; no error if it does
    return wb  # return the resolved path (or None) so callers can check availability


def load_schaefer_parcellation() -> tuple:
    """
    Fetch the Schaefer-100 atlas and extract ROI names (for data processing).

    Returns
    -------
    roi_names : list of str
        100 ROI names sorted by parcel key, excluding Background and Medial_Wall.
    parc_gii  : list of GiftiImage
        Relabelled GIfTI parcellation images [left, right].
    """
    schaefer_annot = fetch_schaefer2018(version="fsaverage5")["100Parcels7Networks"]  # download (or load cached) Schaefer-100 annotation files on the fsaverage5 surface
    parc_gii = annot_to_gifti(schaefer_annot)   # convert FreeSurfer .annot files to GIfTI surface label images
    parc_gii = relabel_gifti(parc_gii)           # relabel parcels so indices are contiguous and consistent across hemispheres

    roi_names_map = {}  # will map integer parcel key → ROI name string
    for hemi in parc_gii:  # iterate over left and right hemisphere GIfTI images
        for lab in hemi.labeltable.labels:  # iterate over every labelled parcel in this hemisphere's label table
            name = lab.label  # extract the human-readable parcel name (e.g. '7Networks_LH_Vis_1')
            if name and "Background" not in name and "Medial_Wall" not in name:  # skip the background label and medial wall, which are not cortical ROIs
                roi_names_map[int(lab.key)] = name  # store the mapping from integer key to ROI name

    roi_names = [roi_names_map[k] for k in sorted(roi_names_map)]  # sort ROIs by their integer key to guarantee a reproducible order

    if len(roi_names) != 100:  # safety check: the Schaefer-100 atlas must yield exactly 100 valid ROIs
        raise ValueError(f"Expected 100 ROI names after filtering, got {len(roi_names)}.")

    return roi_names, parc_gii  # return the ordered name list and the GIfTI images (needed for parcellating volumetric data)


def load_schaefer_surface_parcellation() -> tuple:
    """
    Fetch the Schaefer-100 fsaverage5 parcellation for surface visualization.

    Returns
    -------
    roi_map : dict
        Parcel integer key → ROI name string.
    roi_names : list of str
        ROI names sorted by parcel key.
    hemi_labels : dict
        {'left': array, 'right': array} of vertex-level integer label arrays.
    """
    parc = relabel_gifti(             # relabel to make parcel indices contiguous
        annot_to_gifti(               # convert .annot to GIfTI format
            fetch_schaefer2018(version="fsaverage5")["100Parcels7Networks"]  # download the Schaefer-100 fsaverage5 annotation
        )
    )

    roi_map = {
        int(lab.key): lab.label       # map integer parcel key → ROI name string
        for hemi in parc              # loop over left and right hemisphere images
        for lab in hemi.labeltable.labels  # loop over every entry in the label table
        if lab.label                  # skip entries with empty labels
        and "Background" not in lab.label  # skip the background pseudo-parcel
        and "Medial_Wall" not in lab.label  # skip the medial wall, which is not a cortical region
    }

    roi_names = [roi_map[k] for k in sorted(roi_map)]  # create an ordered list of ROI names, sorted by integer key for reproducibility

    hemi_labels = {
        "left":  np.asarray(parc[0].agg_data()).squeeze().astype(int),  # extract per-vertex integer label array for the left hemisphere
        "right": np.asarray(parc[1].agg_data()).squeeze().astype(int),  # extract per-vertex integer label array for the right hemisphere
    }

    return roi_map, roi_names, hemi_labels  # return the key→name map, ordered names, and per-vertex label arrays for both hemispheres


def load_schaefer_parcellation_for_spins():
    """
    Fetch the Schaefer-100 fsaverage5 parcellation GIfTI for spin permutations.

    Returns
    -------
    parc_gii : list of GiftiImage
        Relabelled GIfTI parcellation images for use with alexander_bloch().
    """
    schaefer_annot = fetch_schaefer2018(version="fsaverage5")["100Parcels7Networks"]  # download (or load cached) the Schaefer-100 fsaverage5 annotation files
    return relabel_gifti(annot_to_gifti(schaefer_annot))  # convert to GIfTI and relabel; the GIfTI format is required by alexander_bloch() for spin permutations


def map_roi_values_to_vertices(
    data_df: pd.DataFrame,
    col: str,
    roi_map: dict,
    hemi_labels: dict,
) -> dict:
    """
    Project ROI-level scalar values onto per-vertex arrays for both hemispheres.

    Parameters
    ----------
    data_df     : DataFrame indexed by ROI name.
    col         : Column name to project.
    roi_map     : Dict mapping parcel integer key → ROI name.
    hemi_labels : Dict with 'left'/'right' vertex-level integer label arrays.

    Returns
    -------
    hemi_data : dict
        {'left': array, 'right': array} of float values; NaN for unlabelled vertices.
    """
    hemi_data = {
        hemi: np.full(labels.shape, np.nan, dtype=float)  # initialise a vertex array filled with NaN (unlabelled vertices remain NaN throughout)
        for hemi, labels in hemi_labels.items()           # create one array per hemisphere, sized to match that hemisphere's vertex count
    }

    for key, roi in roi_map.items():                      # loop over every parcel: integer key and corresponding ROI name
        val = float(data_df.loc[roi, col])                # look up the scalar value for this ROI from the data DataFrame
        for hemi in ["left", "right"]:                    # assign the same value to both hemispheres (parcels exist in only one, so the other won't match)
            hemi_data[hemi][hemi_labels[hemi] == key] = val  # broadcast the ROI value to every vertex whose label matches this parcel key

    for hemi in ["left", "right"]:
        hemi_data[hemi][hemi_labels[hemi] == 0] = np.nan  # overwrite label-0 vertices (background / unlabelled cortex) with NaN so they render transparently

    return hemi_data  # return the populated per-vertex arrays for downstream surface plotting


def save_figure(fig: plt.Figure, name: str, output_dir: str) -> str:
    """
    Save a matplotlib figure to disk and return the output path.

    Non-alphanumeric characters in `name` are replaced with underscores to
    produce a valid filename.

    Parameters
    ----------
    fig        : Matplotlib Figure to save.
    name       : Base name for the file (without extension).
    output_dir : Directory where the figure will be saved.

    Returns
    -------
    path : str
        Full path to the saved PNG file.
    """
    filename = re.sub(r"[^\w\-.]+", "_", name).strip("_") + ".png"  # replace any character that is not a word char, hyphen, or dot with an underscore, then strip leading/trailing underscores and append .png
    path = os.path.join(output_dir, filename)  # construct the full output path by joining the directory and sanitised filename
    fig.savefig(path, dpi=300, bbox_inches="tight")  # save the figure at 300 dpi, cropping to the content bounding box to avoid excess whitespace
    plt.close(fig)   # release the figure from memory to avoid accumulating many open figures during batch processing
    print(f"Saved: {path}")  # log the saved path to the console for progress tracking
    return path  # return the path so callers can collect all saved figure locations


def plot_grid(targets: list, fig_root: str, save_path: str, filename_fmt: str = "{target}.png") -> None:
    """
    Load individual per-target PNG files and arrange them in a 2-column grid.

    Parameters
    ----------
    targets      : Ordered list of target names.
    fig_root     : Directory where the per-target PNGs live.
    save_path    : Full file path for the grid PNG.
    filename_fmt : Format string for per-target filenames; must contain '{target}'.
                   Default: '{target}.png'.
    """
    n_cols = 2  # fix the grid width at two columns; chosen to balance readability and page width
    n_rows = (len(targets) + 1) // n_cols  # compute the number of rows needed, rounding up so all panels fit
    fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 9, n_rows * 7))  # create a figure with one subplot per cell, scaled so each panel is ~9×7 inches
    axes = np.array(axes).flatten()  # flatten the 2-D axes array to a 1-D list for simple sequential indexing

    for ax, target in zip(axes, targets):  # pair each axes object with the corresponding target name
        img_path = os.path.join(fig_root, filename_fmt.format(target=target))  # build the expected path for this target's individual PNG
        if os.path.exists(img_path):         # only attempt to load the image if the file was successfully created
            img = plt.imread(img_path)       # read the PNG into a NumPy array
            ax.imshow(img)                   # display the image in the subplot
            ax.axis("off")                   # hide axis ticks and borders for a clean presentation
            ax.set_title(target)             # label the panel with the target name

    for ax in axes[len(targets):]:  # any surplus axes cells (when the target count is odd) are left blank
        ax.axis("off")              # hide empty cells so they don't appear as white boxes

    fig_grid.tight_layout()  # adjust subplot spacing to prevent label overlap
    fig_grid.savefig(save_path, dpi=300, bbox_inches="tight")  # save the assembled grid at 300 dpi
    plt.close(fig_grid)  # release the figure from memory


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two arrays, returning NaN if either has zero variance.

    Zero-variance arrays cause a division-by-zero in np.corrcoef, so this
    guards against that explicitly.

    Parameters
    ----------
    a, b : array-like
        Input arrays of equal length.

    Returns
    -------
    corr : float
        Pearson r, or NaN if either array is constant.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)  # coerce inputs to float NumPy arrays so std() and corrcoef() behave consistently
    return np.nan if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])  # return NaN for constant arrays (zero std would cause division-by-zero); otherwise return the off-diagonal Pearson r from the 2×2 correlation matrix