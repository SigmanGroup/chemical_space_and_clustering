"""Dimensionality-reduction, clustering, and chemical-space plotting helpers.

The functions in this module operate on molecular descriptor tables or on
two-dimensional embeddings and provide optional static or interactive plots.
"""

import base64
import colorsys
import io
import itertools
from math import isnan
from random import sample

import hdbscan
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from dash import Dash, Input, Output, dcc, html, no_update
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import SilhouetteVisualizer



# Interactive plotting helpers
def n_colors(n):
    """Generate n visually distinct hex colors."""
    return [
        '#%02x%02x%02x' % tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 0.95))
        for i in range(n)
    ]


plotly_marker_symbols = [
    'circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down',
    'triangle-left', 'triangle-right', 'star', 'hexagram', 'hourglass', 'bowtie'
]

matplotlib_markers = ['o', 's', 'D', 'X', 'P', '^', 'v', '<', '>', '*', 'h', 'H', '8']


def mol_to_png(smiles, mol_size=(150, 150), debug=False):
    """Convert a SMILES string to a base64 PNG for Dash tooltips."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Invalid SMILES')
        img = Draw.MolToImage(mol, size=mol_size)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f'data:image/png;base64,{encoded}'
    except Exception as exc:
        if debug:
            print(f"Error converting SMILES '{smiles}': {exc}")
        return None


def scale_size(series, min_size=10, max_size=100, n_points=None):
    """Linearly scale numeric values to marker sizes within a fixed range.

    When ``series`` is ``None``, ``n_points`` constant sizes are returned.
    A constant non-null series is mapped to the midpoint of the size range.
    """
    if series is None:
        if n_points is None:
            raise ValueError('Must provide n_points if series is None')
        return np.full(n_points, min_size)
    series = pd.to_numeric(pd.Series(series), errors='coerce')
    if series.nunique(dropna=True) <= 1:
        return np.full(len(series), (min_size + max_size) / 2)
    return min_size + (series - series.min()) / (series.max() - series.min()) * (max_size - min_size)


def scale_alpha(series, min_alpha=0.5, max_alpha=1.0, n_points=None):
    """Linearly scale numeric values to marker opacities within a fixed range.

    When ``series`` is ``None``, ``n_points`` values equal to ``max_alpha`` are
    returned. A constant non-null series is also mapped to ``max_alpha``.
    """
    if series is None:
        if n_points is None:
            raise ValueError('Must provide n_points if series is None')
        return np.full(n_points, max_alpha)
    series = pd.to_numeric(pd.Series(series), errors='coerce')
    if series.nunique(dropna=True) <= 1:
        return np.full(len(series), max_alpha)
    return min_alpha + (series - series.min()) / (series.max() - series.min()) * (max_alpha - min_alpha)


def map_marker_symbols(series, n_points=None):
    """Map categorical values to Plotly marker symbols, cycling as needed."""
    if series is None:
        if n_points is None:
            raise ValueError('Must provide n_points if series is None')
        return ['circle'] * n_points
    series = pd.Series(series)
    unique_vals = series.dropna().unique()
    symbol_map = {val: plotly_marker_symbols[i % len(plotly_marker_symbols)] for i, val in enumerate(unique_vals)}
    return series.map(symbol_map).fillna('circle')


def map_markers_matplotlib(series, n_points=None):
    """Map categorical values to Matplotlib marker symbols, cycling as needed."""
    if series is None:
        if n_points is None:
            raise ValueError('Must provide n_points if series is None')
        return ['o'] * n_points
    series = pd.Series(series)
    unique_vals = series.dropna().unique()
    symbol_map = {val: matplotlib_markers[i % len(matplotlib_markers)] for i, val in enumerate(unique_vals)}
    return series.map(symbol_map).fillna('o')


def int_scatter_dash(df, x_col, y_col, smiles_col=None, id_col=None, molSize=(150, 150),
                     marker_size=30, marker_symbol='circle', marker_color='blue',
                     marker_alpha=0.7, linewidth=1, lincolor='black', plot_title=None,
                     x_label=None, y_label=None, fig_width=600, fig_height=600,
                     subgroup=False, subgroup_col=None, subgroup_palette='plasma',
                     categorical=False, category_dict=None, categorical_palette=None,
                     overlay_col=None, overlay_color_dict=None, overlay_plot_dict=None,
                     overlay_marker='diamond', size_col=None, min_size=10, max_size=100,
                     alpha_col=None, min_alpha=0.5, max_alpha=1.0,
                     marker_col=None, hover_img_width=None, debug=False,
                     port=8051, run=True):
    """Create a Dash scatter plot with molecular structure + SMILES on hover.

    The hover lookup is based on the unique row position so it remains robust even when
    the scatter plot contains multiple traces, overlays, or repeated IDs.
    """
    import plotly.express as px

    data = df.reset_index(drop=True).copy()
    if x_col not in data.columns or y_col not in data.columns:
        missing = [c for c in [x_col, y_col] if c not in data.columns]
        raise KeyError(f'Missing coordinate columns: {missing}')
    if data.empty:
        raise ValueError('No points to plot')

    if smiles_col is not None:
        data['mol_image'] = data[smiles_col].apply(lambda x: mol_to_png(x, molSize, debug))
    else:
        data['mol_image'] = None
    data['_row_pos'] = np.arange(len(data))

    sizes = scale_size(data[size_col], min_size, max_size) if size_col else np.full(len(data), marker_size)
    alphas = scale_alpha(data[alpha_col], min_alpha, max_alpha) if alpha_col else np.full(len(data), marker_alpha)
    symbols = map_marker_symbols(data[marker_col]) if marker_col else np.full(len(data), marker_symbol)

    fig = go.Figure()

    def add_points(part, color, name=None, showlegend=False, colorscale=None, symbol_override=None):
        idx = part['_row_pos'].to_numpy(dtype=int)
        marker_dict = dict(
            size=sizes[idx],
            symbol=(symbols[idx] if symbol_override is None else symbol_override),
            opacity=(alphas[idx] if symbol_override is None else 1.0),
            color=color,
            line=dict(width=linewidth, color=lincolor)
        )
        if colorscale is not None:
            marker_dict.update(colorscale=colorscale, showscale=True, colorbar=dict(title=str(subgroup_col)))
        fig.add_trace(go.Scatter(
            x=part[x_col],
            y=part[y_col],
            mode='markers',
            customdata=idx.reshape(-1, 1),
            marker=marker_dict,
            name=name,
            showlegend=showlegend,
            hoverinfo='text',
            hovertext=[' '] * len(part)
        ))

    if not subgroup:
        add_points(data, marker_color)
    elif subgroup_col is None:
        raise ValueError('subgroup_col is required when subgroup=True')
    elif categorical:
        groups = list(data[subgroup_col].dropna().unique())
        if category_dict is None:
            if categorical_palette:
                palette = getattr(px.colors.qualitative, categorical_palette)
                palette = (palette * (len(groups) // len(palette) + 1))[:len(groups)]
            else:
                palette = n_colors(len(groups))
            category_dict = dict(zip(groups, palette))
        for group in groups:
            add_points(data.loc[data[subgroup_col].eq(group)], category_dict.get(group, '#888888'),
                       name=str(group), showlegend=True)
        if data[subgroup_col].isna().any():
            add_points(data.loc[data[subgroup_col].isna()], '#888888', name='Missing', showlegend=True)
    else:
        add_points(data, data[subgroup_col], colorscale=subgroup_palette)

    if overlay_col is not None and overlay_color_dict is not None:
        for group, overlay_color in overlay_color_dict.items():
            if overlay_plot_dict is None or overlay_plot_dict.get(group) == 'plot':
                overlay_df = data.loc[data[overlay_col].eq(group)]
                if len(overlay_df) > 0:
                    add_points(overlay_df, overlay_color, name=str(group), showlegend=True,
                               symbol_override=overlay_marker)

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        title=plot_title or '',
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        margin=dict(r=145),
        legend=dict(x=1.05, y=1.0, xanchor='left', yanchor='top')
    )

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id='scatter-graph', figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id='scatter-tooltip', direction='bottom')
    ])

    @app.callback(
        Output('scatter-tooltip', 'show'),
        Output('scatter-tooltip', 'bbox'),
        Output('scatter-tooltip', 'children'),
        Output('scatter-tooltip', 'direction'),
        Input('scatter-graph', 'hoverData')
    )
    def display_hover(hoverData):
        if not hoverData or not hoverData.get('points'):
            return False, no_update, no_update, no_update

        point = hoverData['points'][0]
        print("HOVER DATA:", point)
        row_index = point.get('customdata')
        if row_index is None:
            curve_index = point.get('curveNumber')
            point_index = point.get('pointNumber', point.get('pointIndex'))
            if curve_index is None or point_index is None:
                return False, no_update, no_update, no_update
            trace = fig.data[int(curve_index)]
            if trace.customdata is None:
                return False, no_update, no_update, no_update
            row_index = trace.customdata[int(point_index)]

        row_index = int(np.asarray(row_index).flat[0])
        row = data.iloc[row_index]

        elements = []
        if row['mol_image'] is not None:
            elements.append(html.Img(src=row['mol_image'], style={'width': f'{hover_img_width or molSize[0]}px'}))
        if smiles_col is not None:
            elements.append(html.Div(
                f'SMILES: {row[smiles_col]}',
                style={'overflowWrap': 'anywhere', 'maxWidth': '320px', 'fontSize': '12px', 'marginTop': '6px'}
            ))

        direction = 'bottom' if point['y'] > (data[y_col].min() + data[y_col].max()) / 2 else 'top'
        return True, point['bbox'], html.Div(elements, style={'padding': '5px'}), direction

    if run:
        app.run(debug=debug, use_reloader=False, port=port)
    return app, fig



# Dimensionality reduction
def dim_reduction_umap(features, x_coordinates_label, y_coordinates_label,
                       id_column, smiles_column, excel_filename,
                       n_neighbors, min_dist, n_components, distance_metric,
                       random_state, excel_export=False):
    """Fit UMAP and return coordinates with compound IDs and SMILES strings.

    The supplied ID and SMILES sequences must use the same row order as
    ``features``. Coordinates can optionally be exported to an Excel file.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=distance_metric,
        random_state=random_state
    )
    umap_results = reducer.fit_transform(features)
    umap_df = pd.DataFrame(umap_results, columns=[x_coordinates_label, y_coordinates_label])
    umap_df['id'] = np.asarray(id_column)
    umap_df['smiles'] = np.asarray(smiles_column)
    if excel_export:
        umap_df.to_excel(excel_filename, index=False)
        print(f'UMAP coordinates exported to {excel_filename}')
    return umap_df


def dim_reduction_pca(features, n_components, id_column, smiles_column,
                      export_pc_coords_excel=False, pc_coords_filename=None,
                      export_pc_contrib_excel=False, pc_contrib_filename=None):
    """Fit PCA and return compound coordinates and feature loadings.

    The supplied ID and SMILES sequences must use the same row order as
    ``features``. Coordinate and loading tables can be exported independently.
    """
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(features)
    pca_score = pca.explained_variance_ratio_

    print('Total variance explained by PCs:', round(np.sum(pca_score * 100), 1), '%\n')
    print('Percentage of explained variance per principal component:')
    for i, j in enumerate(pca_score):
        print(f'PC{i + 1}   {j * 100:.1f}%')

    pca_df = pd.DataFrame(pca_results, columns=[f'PC{i + 1}' for i in range(n_components)])
    pca_df['id'] = np.asarray(id_column)
    pca_df['smiles'] = np.asarray(smiles_column)

    feature_name = features.columns if isinstance(features, pd.DataFrame) else [f'Feature {i + 1}' for i in range(features.shape[1])]
    feature_contributions = pd.DataFrame(
        pca.components_.T,
        index=feature_name,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )

    if export_pc_coords_excel:
        pca_df.to_excel(pc_coords_filename, index=False)
        print(f'PCA coordinates exported to {pc_coords_filename}')
    if export_pc_contrib_excel:
        feature_contributions.to_excel(pc_contrib_filename)
        print(f'PCA feature contributions exported to {pc_contrib_filename}')

    return pca_df, feature_contributions


def dim_reduction_tsne(features, x_coordinates_label, y_coordinates_label,
                       id_column, smiles_column, excel_filename,
                       n_components, perplexity, learning_rate,
                       init, random_state, distance_metric,
                       export_excel=False):
    """Fit t-SNE and return coordinates with compound IDs and SMILES strings.

    The supplied ID and SMILES sequences must use the same row order as
    ``features``. Coordinates can optionally be exported to an Excel file.
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        random_state=random_state,
        metric=distance_metric
    )
    tsne_results = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_results, columns=[x_coordinates_label, y_coordinates_label])
    tsne_df['id'] = np.asarray(id_column)
    tsne_df['smiles'] = np.asarray(smiles_column)
    if export_excel:
        tsne_df.to_excel(excel_filename, index=False)
        print(f'tSNE coordinates exported to {excel_filename}')
    return tsne_df


def feat_v_feat(features, id_column, smiles_column, feature_x, feature_y, feature_z):
    """Build a plotting table for two or optionally three descriptors."""
    chem_space_df = features[[feature_x, feature_y] + ([feature_z] if feature_z else [])].copy()
    chem_space_df['id'] = np.asarray(id_column)
    chem_space_df['smiles'] = np.asarray(smiles_column)
    return chem_space_df



# Static plotting helpers
def _legend_inside_opaque(ax, title=None, fontsize=8):
    """Place an opaque legend to the right of an axes and normalize its alpha."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    legend = ax.legend(handles=handles, labels=labels,
                       loc='upper left', bbox_to_anchor=(1.05, 1.0),
                       borderaxespad=0.0, frameon=True, framealpha=0.9,
                       facecolor='white', edgecolor='lightgray',
                       fontsize=fontsize, title=title)
    legend_handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
    for handle in legend_handles:
        try:
            handle.set_alpha(1.0)
        except Exception:
            pass
    return legend


def plot_chemical_space(df, x, y, z=None, input_df=None, fig_size=(6, 6), xlabel=None, ylabel=None, zlabel=None,
                        marker_size=20, marker_color='#7B1B79', alpha=0.7, marker='o',
                        marker_edge_color='black', marker_edge_width=1,
                        title=None, save_plot=False, plot_filename=None,
                        fontsize=12, title_fontsize=14,
                        subgroup=True, subgroup_column_name=None, subgroup_palette=None,
                        category_dict=None, heatmap=False, response_column_name=None, heatmap_palette='plasma',
                        size_col_name=None, min_size=30, max_size=100,
                        alpha_col_name=None, min_alpha=0.3, max_alpha=1.0,
                        marker_col=None, marker_dict=None, plot_order=None,
                        threedim=False):
    """Plot a two- or three-dimensional chemical-space representation.

    Points may be displayed with a single style, colored by categorical
    subgroups, or colored continuously as a heatmap. Optional styling columns
    can be copied from ``input_df``, which must have the same row order as
    ``df``. The function returns the Matplotlib figure and axes objects.
    """
    data = df.copy()
    if input_df is not None:
        for col in [size_col_name, alpha_col_name, marker_col, subgroup_column_name, response_column_name]:
            if col is not None and col not in data.columns and col in input_df.columns:
                data[col] = input_df[col].values

    marker_sizes = scale_size(data[size_col_name], min_size, max_size) if size_col_name else np.full(len(data), marker_size)
    alpha_values = scale_alpha(data[alpha_col_name], min_alpha, max_alpha) if alpha_col_name else np.full(len(data), alpha)

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d') if threedim else fig.add_subplot(111)

    if not subgroup and not heatmap:
        if threedim:
            ax.scatter(data[x], data[y], data[z], s=marker_sizes, c=marker_color, alpha=alpha_values,
                       marker=marker, edgecolor=marker_edge_color, linewidth=marker_edge_width)
        else:
            ax.scatter(data[x], data[y], s=marker_sizes, c=marker_color, alpha=alpha_values,
                       marker=marker, edgecolor=marker_edge_color, linewidth=marker_edge_width)

    elif subgroup and not heatmap:
        if subgroup_column_name is None:
            raise ValueError('subgroup_column_name must be provided when subgroup is True')
        groups = list(pd.Series(data[subgroup_column_name]).dropna().unique())
        if plot_order is not None:
            groups = [g for g in plot_order if g in groups] + [g for g in groups if g not in plot_order]
        if category_dict is None:
            cmap = cm.get_cmap(subgroup_palette or 'tab20', max(len(groups), 1))
            category_dict = {group: cmap(i) for i, group in enumerate(groups)}
        for group in groups:
            subgroup_df = data.loc[data[subgroup_column_name] == group]
            if subgroup_df.empty:
                continue
            idx = subgroup_df.index
            if threedim:
                ax.scatter(subgroup_df[x], subgroup_df[y], subgroup_df[z],
                           s=marker_sizes[idx], c=[category_dict[group]], alpha=alpha_values[idx],
                           edgecolor=marker_edge_color, linewidth=marker_edge_width,
                           marker=marker, label=str(group))
            else:
                ax.scatter(subgroup_df[x], subgroup_df[y],
                           s=marker_sizes[idx], c=[category_dict[group]], alpha=alpha_values[idx],
                           edgecolor=marker_edge_color, linewidth=marker_edge_width,
                           marker=marker, label=str(group))

    elif heatmap and not subgroup:
        if response_column_name is None:
            raise ValueError('response_column_name must be provided when heatmap is True')
        values = pd.to_numeric(data[response_column_name], errors='coerce')
        norm = plt.Normalize(values.min(), values.max())
        cmap = plt.get_cmap(heatmap_palette)
        if marker_col is None:
            if threedim:
                ax.scatter(data[x], data[y], data[z], s=marker_sizes, c=values, cmap=cmap,
                           edgecolor=marker_edge_color, marker=marker,
                           linewidth=marker_edge_width, alpha=alpha_values)
            else:
                ax.scatter(data[x], data[y], s=marker_sizes, c=values, cmap=cmap,
                           edgecolor=marker_edge_color, marker=marker,
                           linewidth=marker_edge_width, alpha=alpha_values)
        else:
            marker_lookup = marker_dict or {k: matplotlib_markers[i % len(matplotlib_markers)] for i, k in enumerate(pd.Series(data[marker_col]).dropna().unique())}
            legend_handles = []
            for marker_group, marker_style in marker_lookup.items():
                part = data.loc[data[marker_col] == marker_group]
                if part.empty:
                    continue
                idx = part.index
                ax.scatter(part[x], part[y], s=marker_sizes[idx], c=values.loc[idx], cmap=cmap,
                           edgecolor=marker_edge_color, marker=marker_style,
                           linewidth=marker_edge_width, alpha=alpha_values[idx])
                legend_handles.append(
                    Line2D([0], [0], marker=marker_style, color='w', markerfacecolor='gray',
                           markersize=8, label=str(marker_group), markeredgecolor=marker_edge_color)
                )
            if legend_handles:
                ax.legend(handles=legend_handles, loc='lower left')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=response_column_name)

    ax.set_xlabel(xlabel or x, fontsize=fontsize)
    ax.set_ylabel(ylabel or y, fontsize=fontsize)
    if threedim:
        ax.set_zlabel(zlabel or z, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    if not threedim:
        ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    if subgroup and not heatmap:
        _legend_inside_opaque(ax, title='Subgroups', fontsize=fontsize)
    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Plot saved as {plot_filename}')
    return fig, ax


def plot_embedding_by_features(embedding_df, feature_df, x, y, feature_names=None,
                               cmap='plasma', fig_size=(6, 6), marker_size=8, alpha=0.7,
                               marker_edge_color=None, type_df=None, overlay_col=None,
                               overlay_color_dict=None, overlay_plot_dict=None, overlay_marker='D',
                               save_dir=None, filename_prefix='', show=True):
    """Create one embedding heatmap per selected descriptor."""
    if feature_names is None:
        feature_names = list(feature_df.columns)
    if len(embedding_df) != len(feature_df):
        raise ValueError('embedding_df and feature_df must contain the same number of rows')
    if type_df is not None and len(embedding_df) != len(type_df):
        raise ValueError('embedding_df and type_df must contain the same number of rows')

    figures = {}
    if save_dir is not None:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for feature in feature_names:
        values = pd.to_numeric(feature_df[feature], errors='coerce')
        fig, ax = plt.subplots(figsize=fig_size)
        norm = mcolors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        ax.scatter(embedding_df[x], embedding_df[y], c=values, cmap=cmap, norm=norm,
                   s=marker_size, alpha=alpha, edgecolor=marker_edge_color)

        if type_df is not None and overlay_col is not None and overlay_color_dict is not None:
            for group, overlay_color in overlay_color_dict.items():
                if overlay_plot_dict is None or overlay_plot_dict.get(group) == 'plot':
                    mask = type_df[overlay_col] == group
                    if np.asarray(mask).any():
                        ax.scatter(embedding_df.loc[mask, x], embedding_df.loc[mask, y],
                                   s=marker_size * 2, c=overlay_color, marker=overlay_marker,
                                   alpha=1.0, edgecolor='none', label=str(group), zorder=10)
            _legend_inside_opaque(ax, title=None, fontsize=8)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(feature)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=feature)
        fig.tight_layout()

        if save_dir is not None:
            from pathlib import Path
            safe = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in str(feature))
            fig.savefig(Path(save_dir) / f'{filename_prefix}{safe}.png', dpi=300, bbox_inches='tight')
        figures[feature] = (fig, ax)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return figures

# Screening helpers
def screen_umap_parameters(features, n_neighbors_values, min_dist_values,
                           metric='euclidean', random_state=42,
                           fig_size=(4, 4), marker_size=12, alpha=0.7,
                           color='#7B1B79', show=True, save_plot=False, plot_filename=None):
    """Qualitatively compare UMAP settings on the same feature matrix."""
    fig, axes = plt.subplots(
        len(n_neighbors_values), len(min_dist_values),
        figsize=(fig_size[0] * len(min_dist_values), fig_size[1] * len(n_neighbors_values)),
        squeeze=False
    )

    for (i, nn), (j, md) in itertools.product(enumerate(n_neighbors_values), enumerate(min_dist_values)):
        reducer = umap.UMAP(n_neighbors=nn, min_dist=md, metric=metric, n_components=2, random_state=random_state)
        xy = reducer.fit_transform(features)
        ax = axes[i, j]
        ax.scatter(xy[:, 0], xy[:, 1], s=marker_size, c=color, alpha=alpha, linewidths=0)
        ax.set_title(f'n_neighbors={nn} | min_dist={md}')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.tight_layout()
    if save_plot and plot_filename:
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


def screen_hdbscan_parameters(df, x, y, min_cluster_sizes, min_samples_values,
                              distance_metric='euclidean', max_cluster_size=0,
                              fig_size=(4, 4), marker_size=5, alpha=0.7,
                              cmap='tab20', save_plot=False, plot_filename=None):
    """Grid-screen HDBSCAN min_cluster_size x min_samples on a fixed 2D embedding."""
    coords = df[[x, y]].to_numpy()
    fig, axes = plt.subplots(
        len(min_cluster_sizes), len(min_samples_values),
        figsize=(fig_size[0] * len(min_samples_values), fig_size[1] * len(min_cluster_sizes)),
        squeeze=False
    )
    results = []

    for i, mcs in enumerate(min_cluster_sizes):
        for j, ms in enumerate(min_samples_values):
            kwargs = dict(min_cluster_size=mcs, min_samples=ms, metric=distance_metric)
            if max_cluster_size not in (None, 0):
                kwargs['max_cluster_size'] = max_cluster_size
            model = hdbscan.HDBSCAN(**kwargs)
            labels = model.fit_predict(coords)
            clustered = labels != -1
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            frac = clustered.mean()

            ax = axes[i, j]
            unique_clusters = sorted(v for v in set(labels) if v != -1)
            palette = plt.get_cmap(cmap, max(len(unique_clusters), 1))
            colors = np.empty((len(labels), 4))
            colors[:] = (0.75, 0.75, 0.75, 0.35)
            for k, cluster in enumerate(unique_clusters):
                colors[labels == cluster] = palette(k)
            ax.scatter(df[x], df[y], c=colors, s=marker_size, alpha=alpha, edgecolor='none')
            ax.set_title(f'mcs={mcs} | ms={ms}\nclusters={n_clusters} | clustered={frac:.0%}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            results.append({
                'min_cluster_size': mcs,
                'min_samples': ms,
                'n_clusters': n_clusters,
                'fraction_clustered': frac
            })

    fig.tight_layout()
    if save_plot and plot_filename:
        fig.savefig(plot_filename, dpi=200, bbox_inches='tight')
    plt.show()
    return pd.DataFrame(results), fig

# Clustering evaluation helpers
def get_hopkins_stat(features):
    """Estimate the Hopkins statistic for cluster tendency.

    Values near 0.5 suggest spatial randomness, while values approaching 1
    suggest stronger clustering tendency. Random sampling makes the result
    stochastic, and ``features`` must support pandas-style row selection.
    """
    d = features.shape[1]
    n = len(features)
    m = max(1, int(0.1 * n))
    if n < 3:
        raise ValueError('Hopkins requires at least 3 rows')

    nbrs = NearestNeighbors(n_neighbors=2).fit(np.asarray(features))
    rand_features = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(np.random.uniform(np.amin(features, axis=0), np.amax(features, axis=0), d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(features.iloc[rand_features[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    hopkins_stat = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(hopkins_stat):
        print(ujd, wjd)
        hopkins_stat = 0
    return hopkins_stat


def silhouette_scores(data, X_coordinate_column, Y_coordinate_column,
                      min_clusters, max_clusters, clustering_function, random_state):
    """Plot silhouette diagnostics across a range of cluster counts.

    Clustering is evaluated on the two specified coordinate columns. The
    supplied clustering function must return ``(fitted_model, data_frame)`` and
    be compatible with Yellowbrick's silhouette visualizer.
    """
    scores = []
    clusters_range = range(min_clusters, max_clusters + 1)

    for n_clusters in clusters_range:
        model, _ = clustering_function(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters)
        coordinates = data[[X_coordinate_column, Y_coordinate_column]]
        clusters = model.predict(coordinates) if hasattr(model, 'predict') else model.labels_
        silhouette_avg = silhouette_score(coordinates, clusters) if 1 < len(set(clusters)) < len(coordinates) else np.nan

        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=ax)
        visualizer.fit(coordinates)

        plt.xlabel('Silhouette Score')
        plt.title(f'Silhouette Score for {n_clusters} Clusters: {silhouette_avg:.2f}')
        plt.show()

        scores.append(silhouette_avg if len(set(clusters)) > 1 else -1)

    plt.plot(clusters_range, scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.show()


def aic_bic_inertia_scores(data, X_coordinate_column, Y_coordinate_column,
                           min_clusters, max_clusters, clustering_function, random_state):
    """Compare available AIC, BIC, or inertia scores over cluster counts.

    AIC and BIC apply to probabilistic models such as GMM, whereas inertia is
    primarily meaningful for centroid-based models such as K-means. Metrics
    unsupported by the fitted model are reported as NaN.
    """
    n_clusters = np.arange(min_clusters, max_clusters + 1)
    aic_scores = []
    bic_scores = []
    inertia_scores = []
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]

    for n in n_clusters:
        model, _ = clustering_function(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters=n)
        bic_scores.append(model.bic(coordinates) if hasattr(model, 'bic') else np.nan)
        aic_scores.append(model.aic(coordinates) if hasattr(model, 'aic') else np.nan)
        inertia_scores.append(model.inertia_ if hasattr(model, 'inertia_') else np.nan)

    plt.figure(figsize=(6, 4))
    if any(np.isfinite(aic_scores)):
        plt.plot(n_clusters, aic_scores, marker='o', linestyle='-', label='AIC')
    if any(np.isfinite(bic_scores)):
        plt.plot(n_clusters, bic_scores, marker='o', linestyle='-', label='BIC')
    if any(np.isfinite(inertia_scores)):
        plt.plot(n_clusters, inertia_scores, marker='o', linestyle='-', label='Inertia')
    plt.xlabel('Number of Components/Clusters')
    plt.ylabel('Score')
    plt.title(f'Clustering Model Evaluation ({clustering_function.__name__})')
    plt.legend(loc='best')
    plt.show()
    return {'aic': aic_scores, 'bic': bic_scores, 'inertia': inertia_scores}

# Clustering models + plotting
def cluster_plots(df, x, y, cluster_column, xlabel, ylabel,
                  cluster_color_map='plasma', marker_size=50, marker_color='black', alpha=0.7,
                  marker='o', marker_edge_color='black', title=None, save_plot=False, plot_filename=None,
                  fontsize=12, title_fontsize=14, subgroup=False, subgroup_column_name=None,
                  subgroup_palette='tab20', subgroup_mark=True, size_col_name=None, fig_size=(6, 6),
                  subgroup_plot_dict=None, subgroup_marker='s', hdbscan_cartoon=False,
                  plot_order=None, scatter_hist=True, color_dictionary=None):
    """Plot labeled clusters and optional highlighted subgroups.

    Cluster colors may be supplied as a mapping or generated from a Matplotlib
    colormap. Optional subgroups are drawn as a second marker layer. The
    function returns the Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    clusters = list(pd.unique(df[cluster_column].dropna()))
    if not clusters:
        raise ValueError('No cluster labels found')

    if isinstance(cluster_color_map, dict):
        cluster_lookup = {cluster: cluster_color_map.get(cluster) for cluster in clusters}
        fallback = plt.get_cmap('tab20')
        for i, cluster in enumerate(clusters):
            if cluster_lookup[cluster] is None:
                cluster_lookup[cluster] = fallback(i % 20)
    else:
        cmap = plt.get_cmap(cluster_color_map or 'plasma')
        cluster_lookup = {cluster: cmap(i / max(len(clusters) - 1, 1)) for i, cluster in enumerate(clusters)}

    point_colors = df[cluster_column].map(cluster_lookup).tolist()
    ax.scatter(df[x], df[y], s=marker_size, marker=marker, c=point_colors,
               alpha=alpha, edgecolor=marker_edge_color, linewidth=1, zorder=1)

    handles = [
        Line2D([], [], marker=marker, color='none', markerfacecolor=cluster_lookup[cluster],
               markeredgecolor=marker_edge_color, label=f'cluster {cluster}')
        for cluster in clusters
    ]

    if subgroup:
        if subgroup_column_name is None:
            raise ValueError('subgroup_column_name required')
        chosen = list(pd.unique(df[subgroup_column_name].dropna())) if subgroup_plot_dict is None else [k for k, v in subgroup_plot_dict.items() if v == 'plot']
        subgroup_cmap = plt.get_cmap(subgroup_palette or 'tab20')
        for i, subgroup_name in enumerate(chosen):
            part = df.loc[df[subgroup_column_name].eq(subgroup_name)]
            if part.empty:
                continue
            subgroup_color = (color_dictionary or {}).get(subgroup_name, subgroup_cmap(i / max(len(chosen) - 1, 1)))
            ax.scatter(part[x], part[y], s=marker_size, marker=subgroup_marker,
                       c=[subgroup_color], alpha=1.0, zorder=2, edgecolor='none', label=str(subgroup_name))
            handles.append(Line2D([], [], marker=subgroup_marker, color='none', markerfacecolor=subgroup_color,
                                  label=str(subgroup_name)))

    legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1.0),
                       title='Clusters & Subgroups' if subgroup else 'Clusters',
                       framealpha=1, facecolor='white', fontsize=fontsize)
    for handle in getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', [])):
        try:
            handle.set_alpha(1)
        except Exception:
            pass

    ax.set_xlabel(xlabel or x, fontsize=fontsize)
    ax.set_ylabel(ylabel or y, fontsize=fontsize)
    if hdbscan_cartoon:
        ax.tick_params(labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    if save_plot and plot_filename:
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    return fig, ax


def kmeans_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters):
    """Fit K-means to two coordinate columns and return labeled copied data."""
    data = data.copy()
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(coordinates)
    data['cluster'] = clusters
    return kmeans, data


def gmm_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, covariance_type='full'):
    """Fit a Gaussian mixture and return labels plus membership probabilities."""
    data = data.copy()
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type=covariance_type)
    gmm.fit(coordinates)
    clusters = gmm.predict(coordinates)
    data['cluster'] = clusters
    soft_labels = gmm.predict_proba(coordinates)
    soft_labels_df = pd.DataFrame(soft_labels, index=data.index, columns=[f'prob_belonging_to_cluster_{i}' for i in range(n_clusters)])
    data_w_probability = pd.concat([data, soft_labels_df], axis=1)
    return gmm, data_w_probability


def hdbscan_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,
                       min_cluster_size, min_samples, distance_metric, max_cluster_size=0,
                       hdbscan_cartoon=False):
    """Fit HDBSCAN to two coordinate columns and return labeled copied data.

    HDBSCAN assigns label ``-1`` to points classified as noise.
    """
    data = data.copy()
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    kwargs = dict(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=distance_metric)
    if max_cluster_size not in (None, 0):
        kwargs['max_cluster_size'] = max_cluster_size
    model = hdbscan.HDBSCAN(**kwargs)
    labels = model.fit_predict(coordinates)
    data['cluster'] = labels
    clustered_points = labels != -1
    print('Fraction of points clustered:', np.sum(clustered_points) / data.shape[0])
    return model, data


def dbscan_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,
                      distance_metric, eps=0.5, min_samples=5):
    """Fit DBSCAN to two coordinate columns and return labeled copied data.

    DBSCAN assigns label ``-1`` to points classified as noise.
    """
    data = data.copy()
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric)
    labels = model.fit_predict(coordinates)
    data['cluster'] = labels
    clustered_points = labels != -1
    print('Fraction of points clustered:', np.sum(clustered_points) / data.shape[0])
    return model, data


def agglomerative_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,
                             n_clusters, distance_metric='euclidean', linkage_type='ward'):
    """Fit agglomerative clustering and return labeled copied data.

    Ward linkage requires the Euclidean distance metric.
    """
    data = data.copy()
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    model = AgglomerativeClustering(n_clusters=n_clusters, metric=distance_metric, linkage=linkage_type)
    labels = model.fit_predict(coordinates)
    data['cluster'] = labels
    return model, data


def elbow_method(data, X_coordinate_column, Y_coordinate_column, min_clusters, max_clusters,
                 clustering_function, random_state):
    """Plot within-cluster sums of squares over a range of cluster counts."""
    coords = data[[X_coordinate_column, Y_coordinate_column]]
    scores = []
    for n in range(min_clusters, max_clusters + 1):
        model, _ = clustering_function(data.copy(), X_coordinate_column, Y_coordinate_column, random_state, n)
        labels = model.predict(coords) if hasattr(model, 'predict') else model.labels_
        center_score = sum(
            ((coords.to_numpy()[labels == k] - coords.to_numpy()[labels == k].mean(axis=0)) ** 2).sum()
            for k in np.unique(labels) if np.any(labels == k)
        )
        scores.append(center_score)
    plt.plot(range(min_clusters, max_clusters + 1), scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squares')
    plt.show()
    return scores


def BIRCH_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,
                     n_clusters, threshold, branching_factor):
    """Fit BIRCH to two coordinate columns and return labeled copied data."""
    data = data.copy()
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    model = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    labels = model.fit_predict(coordinates)
    data['cluster'] = labels
    return model, data
