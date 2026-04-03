"""
Visualize_On_Schaefer.py
------------------------
Surface visualization of working memory associated area's and receptor densities on the
Schaefer-100 cortical parcellation (fsaverage5).

Public API
----------
- visualize_WorkingMemory()
- visualize_receptor_densities()
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from nilearn import datasets, plotting
from sklearn.preprocessing import StandardScaler

from Scripts.utils import (
    load_schaefer_surface_parcellation,
    map_roi_values_to_vertices,
    save_figure,
)


def load_wm_data(wm_csv_path: str) -> pd.DataFrame:
    """Load working memory CSV and set ROI as index."""
    return pd.read_csv(wm_csv_path).set_index("ROI")  # read the WM scores CSV and use the 'ROI' column as the row index so rows can be looked up by brain region name


def build_wm_display_maps(
    hemi_data: dict,
    col: str,
    wm_df: pd.DataFrame,
) -> tuple:
    """
    Build per-hemisphere display arrays and colormap settings for one WM column.

    Parameters
    ----------
    hemi_data : dict
        {'left': array, 'right': array} vertex-level values.
    col : str
        Column name from the WM dataframe.
    wm_df : DataFrame
        Full working memory dataframe (used to derive vmin/vmax).

    Returns
    -------
    hemi_display : dict with 'left' and 'right' display arrays.
    cmap         : matplotlib colormap.
    vmin, vmax   : float colour scale bounds.
    colorbar     : bool, whether to draw a colorbar.
    """
    is_binary = col.startswith("Thr_")  # threshold columns are named 'Thr_<value>' and should be shown with a binary (active / inactive) colormap

    if is_binary:
        cmap = mcolors.ListedColormap(["lightgrey", "red"])  # two-colour map: lightgrey = inactive ROIs (0), red = active ROIs (1)
        vmin, vmax = 0.0, 1.0  # fix the colour scale to the [0, 1] binary range
        colorbar = False  # a colorbar would be uninformative for a two-class binary map

        hemi_display = {}
        for hemi, data in hemi_data.items():  # process each hemisphere separately
            display = np.full(data.shape, np.nan)         # start with all vertices as NaN (will render transparently)
            display[np.isfinite(data)] = 0.0              # set all labelled (non-NaN) vertices to 0 (inactive by default)
            display[np.isfinite(data) & (data > 0)] = 1.0  # overwrite ROIs with a positive WM score to 1 (active)
            hemi_display[hemi] = display

    else:
        cmap = plt.get_cmap("YlOrRd").copy()   # yellow-to-red sequential colormap for the continuous WM score
        cmap.set_under("lightgrey")            # values below vmin (i.e. inactive ROIs) will render in lightgrey rather than the lowest map colour
        vmin = float(np.nanmin(wm_df[col]))    # set the lower colour bound to the minimum WM score across all ROIs
        vmax = float(np.nanmax(wm_df[col]))    # set the upper colour bound to the maximum WM score across all ROIs
        colorbar = True  # a colorbar is needed to interpret the continuous scale

        hemi_display = {}
        for hemi, data in hemi_data.items():  # process each hemisphere separately
            display = data.copy()              # copy the vertex-level data so we can modify it without affecting the original
            inactive = np.isfinite(data) & (data == 0)  # identify parcels that are labelled but have effectively zero WM score (below floating-point noise)
            under_val = vmin - 1e-6  # push inactive parcels slightly below vmin so set_under('lightgrey') kicks in
            display[inactive] = under_val      # assign the under-range value to inactive parcels
            hemi_display[hemi] = display

    return hemi_display, cmap, vmin, vmax, colorbar  # return the modified vertex arrays, colormap, scale bounds, and colorbar flag


def render_wm_surface_figure(
    col: str,
    hemi_display: dict,
    hemi_labels: dict,
    cmap,
    vmin: float,
    vmax: float,
    colorbar: bool,
    fsavg,
) -> plt.Figure:
    """
    Render a 2 by 2 surface figure (left/right by  lateral/medial) for one WM metric.

    Parameters
    ----------
    col          : Column name from the WM dataframe (used for title).
    hemi_display : dict with 'left'/'right' vertex-level display arrays.
    hemi_labels  : dict with 'left'/'right' integer label arrays.
    cmap         : matplotlib colormap.
    vmin, vmax   : colour scale bounds.
    colorbar     : whether to draw a shared colorbar.
    fsavg        : nilearn fsaverage5 surface object.

    Returns
    -------
    fig : matplotlib Figure.
    """
    TITLE_MAP = {
        "WorkingMemory_Score": "Percentage of active vertices",  # for the continuous WM score column
        "Thr_0":     "> 0% active vertices",    # binary map: any ROI with WM > 0
        "Thr_0p001": "> 0.1% active vertices",  # binary map: ROIs exceeding the 0.1% threshold
        "Thr_0p01":  "> 1% active vertices",    # binary map: ROIs exceeding the 1% threshold
        "Thr_0p1":   "> 10% active vertices",   # binary map: ROIs exceeding the 10% threshold
    }

    COLORBAR_LABEL_MAP = {
        "WorkingMemory_Score": "Fraction of active vertices (FDR: 0.01)"  # colorbar label clarifying that the continuous score is FDR-corrected
    }

    VIEWS = [
        ("left",  "lateral", "infl_left",  "sulc_left"),   # left hemisphere, lateral view (outer surface)
        ("left",  "medial",  "infl_left",  "sulc_left"),   # left hemisphere, medial view (inner/midline surface)
        ("right", "lateral", "infl_right", "sulc_right"),  # right hemisphere, lateral view
        ("right", "medial",  "infl_right", "sulc_right"),  # right hemisphere, medial view
    ]

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)  # create a 12×8 inch figure; constrained_layout automatically adjusts spacing for 3-D subplots
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]  # create four 3-D subplots arranged in a 2×2 grid (one per hemisphere/view combination)
    fig.suptitle(TITLE_MAP.get(col, f"Working memory: {col}"), fontsize=16)  # set the overall figure title using the human-readable label if available, otherwise fall back to the column name

    for ax, (hemi, view, surf, bg) in zip(axes, VIEWS):  # pair each subplot with its hemisphere, view direction, surface mesh, and background sulcal depth map
        plotting.plot_surf_stat_map(
            surf_mesh=fsavg[surf],       # the inflated cortical surface mesh (e.g. 'infl_left') for the current hemisphere
            stat_map=hemi_display[hemi], # the per-vertex WM values to colour on this hemisphere
            hemi=hemi,                   # hemisphere identifier ('left' or 'right') for nilearn's plotting orientation
            view=view,                   # camera angle: 'lateral' or 'medial'
            bg_map=fsavg[bg],            # sulcal depth map used as a greyscale background to show cortical folding
            cmap=cmap,                   # the colormap selected in build_wm_display_maps()
            colorbar=False,              # suppress per-subplot colorbars; a single shared colorbar is added below if needed
            vmin=vmin,                   # lower bound of the colour scale
            vmax=vmax,                   # upper bound of the colour scale
            axes=ax,                     # draw into this specific 3-D subplot
        )

        labels = hemi_labels[hemi]       # integer label array for this hemisphere (one value per vertex)
        ids = np.unique(labels)[1:]      # unique parcel IDs, skipping 0 (background / unlabelled vertices)
        plotting.plot_surf_contours(
            surf_mesh=fsavg[surf],       # same surface mesh as above
            roi_map=labels,              # the parcel label array that defines parcel boundaries
            hemi=hemi,                   # hemisphere identifier
            view=view,                   # same camera angle as the stat map so contours align exactly
            levels=ids,                  # draw a contour at each parcel boundary
            colors=["black"] * len(ids), # all parcel boundary lines are drawn in black
            axes=ax,                     # overlay the contours on the same 3-D subplot
        )
        ax.set_title(f"{hemi.capitalize()} {view}")  # label each subplot with the hemisphere and view direction

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))  # create a mappable object with the correct colormap and scale (not directly attached to any axes)
        sm.set_array([])  # required by matplotlib: set a dummy array so the ScalarMappable is considered initialised
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6)  # add a single shared colorbar for all four subplots, shrunk to 60% of the axes height
        cbar.set_label(COLORBAR_LABEL_MAP.get(col, col))  # label the colorbar with the human-readable description or the column name as fallback

    return fig  # return the fully assembled figure for saving by the caller


def visualize_WorkingMemory(wm_csv_path: str, output_dir: str) -> list:
    """
    Surface visualization of working memory metrics on the cortical surface.

    Projects ROI-level values from a CSV file onto the fsaverage5 surface
    using the Schaefer-100 parcellation. For each column a 4-view brain
    figure (left/right by lateral/medial) is generated and saved.

    Parameters
    ----------
    wm_csv_path : str
        Path to CSV file containing ROI-level working memory data.
    output_dir : str
        Directory where generated surface plot images will be saved.

    Returns
    -------
    saved_files : list of str
        Paths to all saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)  # create the output directory if it does not already exist

    wm_df = load_wm_data(wm_csv_path)                                   # load the WM CSV and index rows by ROI name
    roi_map, roi_names, hemi_labels = load_schaefer_surface_parcellation()  # fetch the Schaefer-100 parcellation: parcel key→name map, ordered names, and per-vertex label arrays
    wm_df = wm_df.loc[roi_names]                                         # reorder the WM DataFrame to match the atlas ROI order so vertex mapping is consistent
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")             # download (or load cached) the fsaverage5 inflated surface meshes and sulcal depth maps

    saved_files = []
    for col in wm_df.columns:  # iterate over each WM metric column (continuous score + binary threshold columns)
        hemi_data = map_roi_values_to_vertices(wm_df, col, roi_map, hemi_labels)  # project this column's ROI values onto per-vertex arrays for both hemispheres
        hemi_display, cmap, vmin, vmax, colorbar = build_wm_display_maps(hemi_data, col, wm_df)  # build display-ready vertex arrays and choose the appropriate colormap and scale
        fig = render_wm_surface_figure(col, hemi_display, hemi_labels, cmap, vmin, vmax, colorbar, fsavg)  # assemble the 2×2 four-view brain surface figure
        path = save_figure(fig, col, output_dir)  # save the figure to disk and get the output path
        saved_files.append(path)  # collect the path for the return value


def load_receptor_data_for_viz(
    receptor_csv_path: str,
    receptors: list = None,
) -> pd.DataFrame:
    """
    Load and z-score normalise receptor density data, optionally filtering to a subset.

    Parameters
    ----------
    receptor_csv_path : str
        Path to CSV with ROI-level receptor densities; must contain a 'ROI' column.
    receptors : list of str or None
        Receptor columns to keep. If None, all columns are retained.

    Returns
    -------
    receptor_df : DataFrame, shape (100, n_receptors), z-scored, indexed by ROI.
    """
    receptor_df = pd.read_csv(receptor_csv_path).set_index("ROI")  # load the receptor CSV and use 'ROI' as the row index

    scaler = StandardScaler()
    receptor_df = pd.DataFrame(
        scaler.fit_transform(receptor_df.values),  # z-score every receptor column so all densities are on a comparable unit-free scale
        index=receptor_df.index,                   # preserve the original ROI row labels
        columns=receptor_df.columns,               # preserve the original receptor column names
    )

    if receptors is not None:
        missing = [r for r in receptors if r not in receptor_df.columns]  # identify any requested receptors that are absent from the CSV
        if missing:
            raise ValueError(f"Receptors not found in CSV: {missing}")  # raise early with a clear message rather than silently skipping
        receptor_df = receptor_df[receptors]  # filter to only the requested receptor columns

    return receptor_df  # return the z-scored, optionally filtered receptor DataFrame


def build_receptor_display_maps(
    hemi_data: dict,
    receptor: str,
    receptor_df: pd.DataFrame,
) -> tuple:
    """
    Build per-hemisphere display arrays and colormap settings for one receptor.

    Uses a continuous YlOrRd colormap scaled to the receptor's min/max z-score
    range, with inactive/NaN parcels rendered in lightgrey.

    Parameters
    ----------
    hemi_data   : dict with 'left'/'right' vertex-level float arrays.
    receptor    : column name of the receptor being visualised.
    receptor_df : full z-scored receptor DataFrame (used to derive vmin/vmax).

    Returns
    -------
    hemi_display : dict with 'left' and 'right' display arrays.
    cmap         : matplotlib colormap with lightgrey set for under-range values.
    vmin, vmax   : float colour scale bounds.
    """
    cmap = plt.get_cmap("YlOrRd").copy()           # yellow-to-red sequential colormap for receptor density
    vmin = float(np.nanmin(receptor_df[receptor]))  # lower colour bound: minimum z-scored density for this receptor across all ROIs
    vmax = float(np.nanmax(receptor_df[receptor]))  # upper colour bound: maximum z-scored density for this receptor across all ROIs
    cmap.set_under("lightgrey")                    # parcels with values below vmin (NaN sentinels) will render in lightgrey

    hemi_display = {hemi: data.copy() for hemi, data in hemi_data.items()}  # shallow copy of vertex arrays so any downstream modifications don't affect the originals

    return hemi_display, cmap, vmin, vmax  # return vertex arrays, colormap, and scale bounds for use in the surface figure renderer


def render_receptor_surface_figure(
    receptor: str,
    hemi_display: dict,
    hemi_labels: dict,
    cmap,
    vmin: float,
    vmax: float,
    fsavg,
) -> plt.Figure:
    """
    Render a 2×2 surface figure (left/right × lateral/medial) for one receptor.

    Parameters
    ----------
    receptor     : receptor name used for the figure title and colorbar label.
    hemi_display : dict with 'left'/'right' vertex-level display arrays.
    hemi_labels  : dict with 'left'/'right' integer label arrays.
    cmap         : matplotlib colormap.
    vmin, vmax   : colour scale bounds.
    fsavg        : nilearn fsaverage5 surface object.

    Returns
    -------
    fig : matplotlib Figure.
    """
    VIEWS = [
        ("left",  "lateral", "infl_left",  "sulc_left"),   # left hemisphere, lateral (outer) view
        ("left",  "medial",  "infl_left",  "sulc_left"),   # left hemisphere, medial (inner/midline) view
        ("right", "lateral", "infl_right", "sulc_right"),  # right hemisphere, lateral view
        ("right", "medial",  "infl_right", "sulc_right"),  # right hemisphere, medial view
    ]

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)  # create a 12×8 inch figure with automatic layout adjustment for 3-D subplots
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]  # four 3-D subplots in a 2×2 grid
    fig.suptitle(f"Receptor density: {receptor}", fontsize=16)  # title the figure with the receptor name

    for ax, (hemi, view, surf, bg) in zip(axes, VIEWS):  # pair each subplot with its hemisphere, view, surface mesh, and sulcal depth background
        plotting.plot_surf_stat_map(
            surf_mesh=fsavg[surf],       # inflated cortical surface mesh for the current hemisphere
            stat_map=hemi_display[hemi], # per-vertex z-scored receptor density for this hemisphere
            hemi=hemi,                   # hemisphere identifier for nilearn's camera orientation
            view=view,                   # camera angle: 'lateral' or 'medial'
            bg_map=fsavg[bg],            # sulcal depth map as greyscale background to highlight cortical folding
            cmap=cmap,                   # YlOrRd colormap with lightgrey for under-range values
            colorbar=False,              # suppress per-subplot colorbars; a single shared colorbar is added below
            vmin=vmin,                   # lower bound of the receptor density colour scale
            vmax=vmax,                   # upper bound of the receptor density colour scale
            axes=ax,                     # draw into this 3-D subplot
        )

        labels = hemi_labels[hemi]       # integer parcel label array for this hemisphere
        ids = np.unique(labels)[1:]      # parcel IDs, skipping 0 (background)
        plotting.plot_surf_contours(
            surf_mesh=fsavg[surf],       # same surface mesh so contours align with the stat map
            roi_map=labels,              # parcel label array defining boundary locations
            hemi=hemi,                   # hemisphere identifier
            view=view,                   # same camera angle as the stat map
            levels=ids,                  # draw a contour at each unique parcel ID boundary
            colors=["black"] * len(ids), # all parcel outlines are black for contrast
            axes=ax,                     # overlay on the same 3-D subplot
        )
        ax.set_title(f"{hemi.capitalize()} {view}")  # label each subplot with the hemisphere and view direction

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))  # create a mappable object for the shared colorbar, using the same colormap and scale as the stat maps
    sm.set_array([])  # dummy array assignment required by matplotlib before the ScalarMappable can be used with fig.colorbar()
    fig.colorbar(sm, ax=axes, shrink=0.6).set_label(f"{receptor} density (z-scored)")  # add one shared colorbar across all subplots and label it with the receptor name and units

    return fig  # return the completed figure for saving by the caller


def visualize_receptor_densities(
    receptor_csv_path: str,
    output_dir: str,
    receptors: list = None,
) -> list:
    """
    Surface visualization of receptor densities on the cortical surface.

    Projects ROI-level receptor density values onto the fsaverage5 cortical
    surface using the Schaefer-100 parcellation. For each selected receptor,
    a 4-view brain figure (left/right × lateral/medial) is generated and saved.

    Parameters
    ----------
    receptor_csv_path : str
        Path to CSV file containing ROI-level receptor density data.
        Must contain a 'ROI' index column.
    output_dir : str
        Directory where generated surface plot images will be saved.
    receptors : list of str, optional
        Receptor columns to visualise. If None, all columns are used.

    Returns
    -------
    saved_files : list of str
        Paths to all saved figure files.
    """
    os.makedirs(output_dir, exist_ok=True)  # create the output directory if it does not already exist

    receptor_df = load_receptor_data_for_viz(receptor_csv_path, receptors)          # load, z-score, and optionally filter the receptor density DataFrame
    roi_map, roi_names, hemi_labels = load_schaefer_surface_parcellation()           # fetch Schaefer-100 parcellation data: key→name map, ordered names, per-vertex label arrays
    receptor_df = receptor_df.loc[roi_names]                                         # reorder rows to match the atlas ROI order so vertex mapping is consistent
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")                         # download (or load cached) the fsaverage5 surface meshes and sulcal depth maps

    saved_files = []
    for receptor in receptor_df.columns:  # iterate over each receptor column to produce one figure per receptor
        hemi_data = map_roi_values_to_vertices(receptor_df, receptor, roi_map, hemi_labels)  # project this receptor's ROI values onto per-vertex arrays for both hemispheres
        hemi_display, cmap, vmin, vmax = build_receptor_display_maps(hemi_data, receptor, receptor_df)  # build display-ready vertex arrays and choose the colormap and scale bounds
        fig = render_receptor_surface_figure(receptor, hemi_display, hemi_labels, cmap, vmin, vmax, fsavg)  # assemble the 2×2 four-view brain surface figure
        path = save_figure(fig, receptor, output_dir)  # save the figure to disk and capture the output path
        saved_files.append(path)  # collect all saved paths for the return value