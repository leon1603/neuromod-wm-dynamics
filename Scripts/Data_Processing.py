"""
Data_Processing.py
------------------
Functions for binarising a Neurosynth NIfTI map at the vertex level and
parcellating neuroreceptor, T1w/T2w, and working memory data onto the
Schaefer-100 atlas.

Public API
----------
- binarize_nifti()
- process_neuro_receptor_data()
"""

import os
import subprocess

import pandas as pd
import nibabel as nib
from neuromaps import transforms
from neuromaps.parcellate import Parcellater

from Scripts.utils import setup_environment, load_schaefer_parcellation

def binarize_nifti(input_nii: str, output_nii: str, threshold: float) -> None:
    """
    Binarize a NIfTI volume using Connectome Workbench with a custom threshold.

    Applies a threshold (x > threshold) to the input volume, producing a
    binary output where values above the threshold are set to 1 and all other
    values are set to 0.

    Parameters
    ----------
    input_nii : str
        Path to the input NIfTI file.
    output_nii : str
        Path where the binarized NIfTI file will be saved.
    threshold : float
        Threshold value applied to the input data (x > threshold).
    """
    wb_view_path = "/Applications/wb_view.app/Contents/MacOS"               # path to the Connectome Workbench directory
    os.environ["PATH"] = f"{wb_view_path}:" + os.environ.get("PATH", "")    # prepend the Workbench directory to PATH so wb_command is discoverable by subprocess

    command = [
        "wb_command",               # the Connectome Workbench command-line tool
        "-volume-math",             # subcommand: evaluate a mathematical expression voxel-by-voxel on a volume
        f"(x > {threshold})",       # the expression: 1 where the voxel value exceeds the threshold, 0 elsewhere
        output_nii,                 # path where the resulting binary NIfTI will be written
        "-var", "x", input_nii,     # declare the variable x and bind it to the input NIfTI file
    ]

    subprocess.run(command, check=True)  # execute the command and raise an error if wb_command returns a non-zero exit code


def load_receptor_data(
    receptor_names_path: str,
    receptor_data_path: str,
    roi_names: list,
) -> pd.DataFrame:
    """
    Load PET receptor names and density data, label with ROI names.

    Parameters
    ----------
    receptor_names_path : str
        Path to a CSV file containing PET receptor names (one per row).
    receptor_data_path : str
        Path to a CSV file of parcellated receptor density data (100 rows × N receptors).
    roi_names : list of str
        100 ROI names from the Schaefer atlas.

    Returns
    -------
    receptor_df : DataFrame indexed by ROI.
    """
    receptor_names = pd.read_csv(receptor_names_path).iloc[:, 0].values  # read only the first column, which contains one receptor/transporter name per row, as a NumPy array

    receptor_df = pd.read_csv(receptor_data_path, header=None)  # load the numeric density matrix (100 ROIs × N receptors); no header row in this file
    receptor_df.columns = receptor_names  # assign the receptor names as column headers

    receptor_df = receptor_df.reset_index(drop=True)  # ensure the row index is a clean 0-based integer range before reassigning
    receptor_df.index = roi_names    # label rows with the 100 Schaefer ROI names so data can later be aligned by ROI
    receptor_df.index.name = "ROI"  # name the index column 'ROI' for consistent downstream merging and CSV export

    return receptor_df  # return the fully labelled receptor density DataFrame


def load_t1wt2w_data(T1wT2W_df_path: str, roi_names: list) -> pd.DataFrame:
    """
    Load T1w/T2w data and label with ROI names.

    Parameters
    ----------
    T1wT2W_df_path : str
        Path to a CSV file containing T1w/T2w myelination values (100 ROIs).
    roi_names : list of str
        100 ROI names from the Schaefer atlas.

    Returns
    -------
    t1wt2w_df : DataFrame indexed by ROI.
    """
    raw = pd.read_csv(T1wT2W_df_path)           # load the CSV
    t1wt2w_df = raw.iloc[:, 1:].copy()          # drop the first column (assumed to be a stale row-index column) and copy to avoid modifying the original
    t1wt2w_df.index = roi_names                 # assign the 100 Schaefer ROI names as the row index
    t1wt2w_df.index.name = "ROI"               # name the index 'ROI' for consistent downstream use
    return t1wt2w_df  # return the myelination DataFrame indexed by ROI name


def parcellate_wm_map(
    wm_map_path: str,
    parc_gii,
    roi_names: list,
) -> pd.DataFrame:
    """
    Transform a MNI152 working memory NIfTI to fsaverage and parcellate.

    Parameters
    ----------
    wm_map_path : str
        Path to the NIfTI file containing the working memory map in MNI152 space.
    parc_gii : list of GiftiImage
        Schaefer-100 parcellation GIfTI images.
    roi_names : list of str
        100 ROI names from the Schaefer atlas.

    Returns
    -------
    wm_parc : DataFrame, shape (100, 1), column 'WorkingMemory_Score', indexed by ROI.
    """
    wm_mni = nib.load(wm_map_path)  # load the Neurosynth working memory NIfTI volume in MNI152 space
    wm_fs = transforms.mni152_to_fsaverage(wm_mni, fsavg_density="10k", method="linear")  # resample the volumetric map onto the fsaverage surface at 10k vertex density using linear interpolation

    parcellater = Parcellater(parc_gii, "fsaverage", resampling_target="data").fit()  # create a parcellater that will average vertex values within each of the 100 Schaefer parcels; 'data' resampling aligns the parcellation to the data resolution
    wm_parc_data = parcellater.transform(wm_fs, "fsaverage")  # apply the parcellater: average vertex-level working memory values within each parcel, yielding a (1, 100) array

    wm_parc = pd.DataFrame(wm_parc_data, columns=["WorkingMemory_Score"], index=roi_names)  # wrap the parcellated values in a DataFrame with the 100 ROI names as the index and a descriptive column name
    wm_parc.index.name = "ROI"  # name the index 'ROI' for consistent downstream merging and CSV export

    return wm_parc  # return the continuous working memory score per ROI


def apply_wm_thresholds(wm_parc: pd.DataFrame, thresholds: list) -> pd.DataFrame:
    """
    Append binary threshold columns to the continuous WM parcellation.

    Each threshold t produces a column 'Thr_<t>' where 1 = score > t.

    Parameters
    ----------
    wm_parc    : DataFrame with a 'WorkingMemory_Score' column.
    thresholds : list of float

    Returns
    -------
    wm_binary_df : DataFrame with continuous score + one binary column per threshold.
    """
    wm_binary_df = wm_parc.copy()  # copy to avoid mutating the input DataFrame

    for thr in thresholds:  # iterate over each requested threshold value
        col_name = f"Thr_{str(thr).replace('.', 'p')}"  # build a filesystem-safe column name by replacing the decimal point with 'p' (e.g. 0.01 → 'Thr_0p01')
        wm_binary_df[col_name] = (wm_parc["WorkingMemory_Score"] > thr).astype(int)  # create a binary column: 1 where the continuous WM score exceeds the threshold, 0 otherwise

    return wm_binary_df  # return the DataFrame containing both the continuous score and all binary threshold columns


def save_outputs(
    output_directory_path: str,
    receptor_df: pd.DataFrame,
    t1wt2w_df: pd.DataFrame,
    t1wt2w_receptor_df: pd.DataFrame,
    wm_binary_df: pd.DataFrame,
    receptor_df_name: str,
    T1wT2W_df_name: str,
    T1wT2W_receptor_df_name: str,
    wm_parc_name: str,
    thresholds: list,
) -> None:
    """Write all output DataFrames to CSV."""
    receptor_df.to_csv(os.path.join(output_directory_path, receptor_df_name))           # save the receptor-only density DataFrame (100 ROIs × 19 receptors)
    t1wt2w_df.to_csv(os.path.join(output_directory_path, T1wT2W_df_name))               # save the standalone T1w/T2w myelination DataFrame
    t1wt2w_receptor_df.to_csv(os.path.join(output_directory_path, T1wT2W_receptor_df_name))  # save the combined T1w/T2w + receptor DataFrame used as the predictor matrix in some analyses
    wm_binary_df.to_csv(os.path.join(output_directory_path, wm_parc_name))              # save the working memory DataFrame (continuous score + binary threshold columns)

    print(f"WM continuous + binary thresholds saved in: {wm_parc_name}")  # confirm which file holds the WM outputs
    print(f"Receptor density maps saved in: {receptor_df_name}")
    print(f"T1w/T2w saved in: {T1wT2W_df_name}")
    print(f"Receptor density maps + T1w/T2w saved in: {T1wT2W_receptor_df_name}")


def process_neuro_receptor_data(
    wb_view_path: str,
    wm_map_path: str,
    receptor_names_path: str,
    receptor_data_path: str,
    T1wT2W_df_path: str,
    output_directory_path: str,
    receptor_df_name: str,
    T1wT2W_receptor_df_name: str,
    wm_parc_name: str,
    T1wT2W_df_name: str,
    thresholds: list = None,
) -> None:
    """
    Preprocessing pipeline for parcellating and aligning neuroreceptor and
    working memory data.

    Combines PET-derived receptor densities, T1w/T2w myelination data, and a
    binarised Neurosynth working memory map into a common Schaefer-100
    parcellation. All outputs are saved as labelled CSV files indexed by ROI.

    Parameters
    ----------
    wb_view_path : str
        Path to the wb_command binary directory.
    wm_map_path : str
        Path to a NIfTI file containing the working memory map in MNI152 space.
    receptor_names_path : str
        Path to a CSV file containing PET receptor names (one per row).
    receptor_data_path : str
        Path to a CSV file of parcellated receptor density data (100 rows × N receptors).
    T1wT2W_df_path : str
        Path to a CSV file containing T1w/T2w myelination values across the 100 Schaefer ROIs.
    output_directory_path : str
        Directory where all output CSV files will be written.
    receptor_df_name : str
        Filename for the saved receptor dataframe.
    T1wT2W_receptor_df_name : str
        Filename for the combined T1w/T2w + receptor dataframe.
    wm_parc_name : str
        Filename for the parcellated working memory dataframe.
    T1wT2W_df_name : str
        Filename for the standalone T1w/T2w dataframe.
    thresholds : list of float, optional
        Thresholds applied to WM scores to generate binary ROI vectors.
    """
    if thresholds is None:  # default to an empty list so apply_wm_thresholds loops over nothing and only the continuous score is saved
        thresholds = []

    # 1. Setup
    setup_environment(wb_view_path, output_directory_path)  # add wb_command to PATH and create the output directory

    # 2. Parcellation atlas
    roi_names, parc_gii = load_schaefer_parcellation()  # download and parse the Schaefer-100 atlas: get ordered ROI names and GIfTI parcellation images

    # 3. Load data
    receptor_df = load_receptor_data(receptor_names_path, receptor_data_path, roi_names)  # load PET receptor densities and label rows with the 100 Schaefer ROI names
    t1wt2w_df = load_t1wt2w_data(T1wT2W_df_path, roi_names)                              # load T1w/T2w myelination values and label rows with the 100 Schaefer ROI names
    t1wt2w_receptor_df = pd.concat([t1wt2w_df, receptor_df], axis=1)                     # horizontally concatenate myelination and receptor DataFrames to form the full predictor matrix

    # 4. Working memory map
    wm_parc = parcellate_wm_map(wm_map_path, parc_gii, roi_names)  # resample the MNI152 Neurosynth WM map to fsaverage and average within each Schaefer parcel
    wm_binary_df = apply_wm_thresholds(wm_parc, thresholds)         # append binary columns for each threshold (e.g. score > 0.01) to the continuous WM score DataFrame

    # 5. Save
    save_outputs(
        output_directory_path,                                   # write all four DataFrames to this directory
        receptor_df, t1wt2w_df, t1wt2w_receptor_df, wm_binary_df,  # the four output DataFrames
        receptor_df_name, T1wT2W_df_name, T1wT2W_receptor_df_name,  # filenames for the first three outputs
        wm_parc_name, thresholds,                               # filename for the WM output and the threshold list for logging
    )