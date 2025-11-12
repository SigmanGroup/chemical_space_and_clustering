import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem
import hdbscan

from sklearn.neighbors import NearestNeighbors
from math import isnan
from numpy.random import uniform

from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
from random import sample

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, DBSCAN, Birch, OPTICS, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score, pairwise_distances, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from random import sample
from math import isnan
from numpy.random import uniform
from sklearn.decomposition import PCA,NMF

from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
import importlib
from sklearn.cluster import AgglomerativeClustering, Birch 
import itertools
from sklearn.cluster import Birch

import io
import base64
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib import colors as mcolors

from yellowbrick.cluster import SilhouetteVisualizer

### --- interactive plotting functions --- ###
#cycle through n color so each cluster is a unique color
def n_colors(n): #s = 0.65 is color intensity (0 gray, 1 full color), v = 0.95 is brightness (0 black, 1 full color)
    return ['#%02x%02x%02x' % tuple(int(255*c) for c in colorsys.hsv_to_rgb(i/n, 0.65, 0.95))
            for i in range(n)] #'#%02x%02x%02x' is hex formatting

plotly_marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down',
    'triangle-left', 'triangle-right', 'star', 'hexagram', 'hourglass', 'bowtie']

### --- convert smiles to png image encoded in base64 --- ###
def mol_to_png(smiles, molSize=(150,150), debug=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        img = Draw.MolToImage(mol, size=molSize)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        if debug:
            print(f"Error converting SMILES '{smiles}': {e}")
        return None
    
### --- helper functions to scale marker style by dataframe --- ###
def scale_size(series, min_size=10, max_size=100, n_points=None):
    if series is None:
        if n_points is None:
            raise ValueError("Must provide n_points if series is None")
        return np.array([min_size]*n_points)
    s = np.array(series)
    return min_size + (s - s.min()) / (s.max() - s.min() + 1e-9) * (max_size - min_size)

def scale_alpha(series, min_alpha=0.5, max_alpha=1.0, n_points=None):
    if series is None:
        if n_points is None:
            raise ValueError("Must provide n_points if series is None")
        return np.array([max_alpha]*n_points)
    a = np.array(series)
    return min_alpha + (a - a.min()) / (a.max() - a.min() + 1e-9) * (max_alpha - min_alpha)

def map_marker_symbols(series, n_points=None):
    if series is None:
        if n_points is None:
            raise ValueError("Must provide n_points if series is None")
        return ['circle']*n_points
    unique_vals = series.dropna().unique()
    symbol_map = {val: plotly_marker_symbols[i % len(plotly_marker_symbols)] for i, val in enumerate(unique_vals)}
    return series.map(symbol_map).fillna('circle')

# reference: https://dash.plotly.com/dash-core-components/tooltip
def int_scatter_dash(df, x_col, y_col, smiles_col=None, id_col=None,molSize=(150,150), 
                     marker_size=30, marker_symbol="circle", marker_color='blue',
                     marker_alpha=0.7, linewidth=1, lincolor='black', plot_title=None,
                     x_label=None, y_label=None, fig_width=600, fig_height=600,
                     subgroup=False, subgroup_col=None, subgroup_palette='plasma',
                     categorical=False, category_dict=None, categorical_palette=None,
                     size_col=None, min_size=10, max_size=100,
                     alpha_col=None, min_alpha=0.5, max_alpha=1.0,
                     marker_col=None, hover_img_width=None, debug=False):
    
    if smiles_col:
        df['mol_image'] = df[smiles_col].apply(lambda smi: mol_to_png(smi, molSize=molSize, debug=debug))
    else:
        df['mol_image'] = [None]*len(df)
    
    sizes = scale_size(df[size_col] if size_col else None, min_size, max_size, n_points=len(df))
    alphas = scale_alpha(df[alpha_col] if alpha_col else None, min_alpha, max_alpha, n_points=len(df))
    symbols = map_marker_symbols(df[marker_col] if marker_col else None, n_points=len(df))
    
    if hover_img_width is None:
        hover_img_width = molSize[0]
    
    fig = go.Figure()
    if not subgroup: #no subgroup normal plot
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(
                size=sizes if size_col else marker_size,
                color=marker_color,
                symbol=symbols if marker_col else marker_symbol,
                opacity=alphas if alpha_col else marker_alpha,
                line=dict(width=linewidth, color=lincolor)
            ),
            hoverinfo='none'
        ))
    else:
        if subgroup_col is None:
            raise ValueError("Please specify a column name for `subgroup_col` when subgroup=True")
        
        if categorical: # using discrete color palette
            if category_dict is None:
                if categorical_palette is None:
                    color_list = n_colors(len(df[subgroup_col].dropna().unique()))
                else:
                    import plotly.express as px
                    colors = px.colors.qualitative.__getattribute__(categorical_palette)
                    unique_groups = df[subgroup_col].dropna().unique()
                    color_list = (colors * ((len(unique_groups) // len(colors)) + 1))[:len(unique_groups)]
                category_dict = dict(zip(df[subgroup_col].dropna().unique(), color_list))
            
            for group, color in category_dict.items():
                subgroup_df = df[df[subgroup_col] == group]
                fig.add_trace(go.Scatter(
                    x=subgroup_df[x_col],
                    y=subgroup_df[y_col],
                    mode='markers',
                    marker=dict(
                        size=sizes[df[subgroup_col] == group] if size_col else marker_size,
                        color=color,
                        symbol=symbols[df[subgroup_col] == group] if marker_col else marker_symbol,
                        opacity=alphas[df[subgroup_col] == group] if alpha_col else marker_alpha,
                        line=dict(width=linewidth, color=lincolor)
                    ),
                    name=str(group),
                    hoverinfo='none'
                ))
        else: # using cont color palette
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(
                    size=sizes if size_col else marker_size,
                    color=df[subgroup_col],
                    colorscale=subgroup_palette,
                    symbol=symbols if marker_col else marker_symbol,
                    opacity=alphas if alpha_col else marker_alpha,
                    line=dict(width=linewidth, color=lincolor),
                    colorbar=dict(title=subgroup_col)
                ),
                hoverinfo='none'
            ))
    
    fig.update_layout(
        xaxis=dict(range=[df[x_col].min()-1, df[x_col].max()+1]),
        yaxis=dict(range=[df[y_col].min()-1, df[y_col].max()+1]),
        width=fig_width,
        height=fig_height,
        title=plot_title if plot_title else "",
        xaxis_title=x_label if x_label else x_col,
        yaxis_title=y_label if y_label else y_col
    )
    
    # dash app for interactive hover
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="scatter-graph", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="scatter-tooltip", direction="bottom")
    ])
    
    @app.callback(
        Output("scatter-tooltip", "show"),
        Output("scatter-tooltip", "bbox"),
        Output("scatter-tooltip", "children"),
        Output("scatter-tooltip", "direction"),
        Input("scatter-graph", "hoverData")
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update, no_update

        idx = hoverData['points'][0]['pointIndex']
        bbox = hoverData['points'][0]['bbox']
        y_val = hoverData['points'][0]['y']
        direction = "bottom" if y_val > (df[y_col].max() + df[y_col].min()) / 2 else "top"

        img_src = df['mol_image'].iloc[idx]
        id_val = df[id_col].iloc[idx]  

        children = html.Div(
            [
                html.Img(
                    src=img_src,
                    style={
                        "width": f"{hover_img_width}px",
                        "display": "block",
                        "margin-bottom": "5px",
                        "border": "1px solid #ccc",
                        "border-radius": "4px"
                    }
                ),
                html.Div(
                    f"ID: {id_val}",
                    style={
                        "text-align": "center",
                        "font-weight": "bold",
                        "font-size": "12px"
                    }
                )
            ],
            style={"padding": "5px"} 
        )

        return True, bbox, children, direction


    app.run(debug=True, use_reloader=False, port=8051)


### --- dimensionality reduction functions --- ###
def dim_reduction_umap(features,x_coordinates_label, y_coordinates_label,
                       id_column, smiles_column, excel_filename,
                       n_neighbors, min_dist, n_components, distance_metric,
                       random_state, excel_export=False):
    
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        metric=distance_metric,
                        random_state=random_state)
    
    umap_results = reducer.fit_transform(features)
    umap_df = pd.DataFrame(umap_results, columns=[x_coordinates_label, y_coordinates_label])

    umap_df['id'], umap_df['smiles'] = id_column, smiles_column

    if excel_export:
        umap_df.to_excel(excel_filename, index=False)
        print(f'UMAP coordinates exported to {excel_filename}')

    return umap_df

def dim_reduction_pca(features, n_components, id_column, smiles_column,
                      export_pc_coords_excel=False, pc_coords_filename=None,
                      export_pc_contrib_excel=False, pc_contrib_filename=None):
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(features)
    pca_score = pca.explained_variance_ratio_

    print('Total variance explained by PCs:', round(np.sum(pca_score * 100), 1), '%\n')
    print("Percentage of explained variance per principal component:")
    for i, j in enumerate(pca_score):
        print(f"PC{i+1}   {j * 100:.1f}%")

    pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['id'], pca_df['smiles'] = id_column, smiles_column

    feature_name = features.columns if isinstance(features, pd.DataFrame) else [f'Feature {i+1}' for i in range(features.shape[1])]
    feature_contributions = pd.DataFrame(pca.components_.T,
                                          index=feature_name, 
                                          columns=[f'PC{i+1}' for i in range(n_components)])
    if export_pc_coords_excel:
        pca_df.to_excel(pc_coords_filename, index=False)
        print(f'PCA coordinates exported to {pc_coords_filename}')

    if export_pc_contrib_excel:
        feature_contributions.to_excel(pc_contrib_filename)
        print(f'PCA feature contributions exported to {pc_contrib_filename}')

    return pca_df, feature_contributions

def dim_reduction_tsne (features, x_coordinates_label, y_coordinates_label,
                        id_column, smiles_column, excel_filename,
                        n_components, perplexity, learning_rate,
                        init, random_state, distance_metric,
                        export_excel=False):
    
    tsne = TSNE (n_components=n_components, perplexity=perplexity,
                 learning_rate=learning_rate, init=init,
                 random_state=random_state, metric=distance_metric)
    
    tsne_results = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_results, columns=[x_coordinates_label, y_coordinates_label])
    tsne_df['id'], tsne_df['smiles'] = id_column, smiles_column

    if export_excel:
        tsne_df.to_excel(excel_filename, index=False)
        print (f'tSNE coordinates exported to {excel_filename}')

    return tsne_df

def feat_v_feat (features, id_column, smiles_column, 
                 feature_x, feature_y, feature_z):

    chem_space_df = features[[feature_x, feature_y] + ([feature_z] if feature_z else [])].copy()
    chem_space_df['id'], chem_space_df['smiles'] =  id_column, smiles_column

    return chem_space_df


### --- Cleaner + generalized plotting functions, not interactive --- ###
matplotlib_markers = ['o', 's', 'D', 'X', 'P', '^', 'v', '<', '>', '*', 'h', 'H', '8']

def map_markers_matplotlib(series, n_points=None):
    if series is None:
        if n_points is None:
            raise ValueError("Must provide n_points if series is None")
        return ['circle']*n_points
    unique_vals = series.dropna().unique()
    symbol_map = {val: matplotlib_markers[i % len(matplotlib_markers)] for i, val in enumerate(unique_vals)}
    return series.map(symbol_map).fillna('circle')


def plot_chemical_space(df, x, y, z=None, input_df=None, fig_size=(6,6), xlabel=None, ylabel=None, zlabel=None,
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
    
    # set up data for size, alpha, and marker type if specified
    if size_col_name and size_col_name not in df.columns:
        df[size_col_name] = input_df[size_col_name]
    if alpha_col_name and alpha_col_name not in df.columns:
        df[alpha_col_name] = input_df[alpha_col_name]
    if marker_col and marker_col not in df.columns:
        df[marker_col] = input_df[marker_col]
    
    marker_sizes = scale_size(df[size_col_name] if size_col_name else None, min_size, max_size, n_points=len(df))
    alpha_values = scale_alpha(df[alpha_col_name] if alpha_col_name else None, min_alpha, max_alpha, n_points=len(df))
    
    if marker_col and not marker_dict:
        markers = map_markers_matplotlib(df[marker_col] if marker_col else None, n_points=len(df))
    elif marker_col and marker_dict:
        markers = df[marker_col].map(marker_dict).fillna('o')

    fig = plt.figure(figsize=fig_size)
    if threedim:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # plot simple chemical space
    if not subgroup and not heatmap:
        if threedim:
            ax.scatter(df[x], df[y], df[z], s=marker_size, marker=marker, c=marker_color, alpha=alpha,
                       edgecolor=marker_edge_color, linewidth=marker_edge_width)
        else:
            ax.scatter(df[x], df[y], s=marker_size, marker=marker, c=marker_color, alpha=alpha,
                       edgecolor=marker_edge_color, linewidth=marker_edge_width)
    
    # plot chemical space with subgroups
    elif subgroup and not heatmap:
        if subgroup_column_name is None:
            raise ValueError("subgroup_column_name must be provided when subgroup is True")
        subgroup_column = input_df[subgroup_column_name]
        df[subgroup_column_name] = subgroup_column
        # use a matplotlib palette for subgroups
        if category_dict is None:
            if subgroup_palette is None:
                colors = cm.get_cmap('tab20c', len(df[subgroup_column_name].dropna().unique()))
                color_list = [colors(i) for i in range(len(df[subgroup_column_name].dropna().unique()))]
            else:
                colors = cm.get_cmap(subgroup_palette, len(df[subgroup_column_name].dropna().unique()))
                color_list = [colors(i) for i in range(len(df[subgroup_column_name].dropna().unique()))]
            category_dict = dict(zip(df[subgroup_column_name].dropna().unique(), color_list))
        # or using a dictionary put in by user
        for group, color in category_dict.items():
            subgroup_df = df[df[subgroup_column_name] == group]
            if threedim:
                ax.scatter(subgroup_df[x], subgroup_df[y], subgroup_df[z],
                           s=marker_sizes[subgroup_df.index] if size_col_name else marker_size,
                           c=color,
                           alpha=alpha_values[subgroup_df.index] if alpha_col_name else alpha,
                           edgecolor=marker_edge_color,
                           linewidth=marker_edge_width,
                           marker=markers[subgroup_df.index] if marker_col else marker,
                           label=str(group))
            else:
                ax.scatter(subgroup_df[x], subgroup_df[y],
                           s=marker_sizes[subgroup_df.index] if size_col_name else marker_size,
                           c=color,
                           alpha=alpha_values[subgroup_df.index] if alpha_col_name else alpha,
                           edgecolor=marker_edge_color,
                           linewidth=marker_edge_width,
                           marker=markers[subgroup_df.index] if marker_col else marker,
                           label=str(group))
    
    #plot heatmap of chemical space 
    elif heatmap and not subgroup:
        if response_column_name is None:
            raise ValueError("response_column_name must be provided when heatmap is True")
        elif response_column_name not in df.columns:
            response_column = input_df[response_column_name]
            df[response_column_name] = response_column
        norm = plt.Normalize(df[response_column_name].min(), df[response_column_name].max())
        scalarmap = plt.cm.ScalarMappable(cmap=heatmap_palette, norm=norm)
        if threedim:
            ax.scatter(df[x], df[y], df[z],
                       s=marker_sizes if size_col_name else marker_size,
                       c=df[response_column_name],
                       cmap=heatmap_palette,
                       edgecolor=marker_edge_color,
                       marker=marker,
                       linewidth=marker_edge_width,
                       alpha=alpha_values if alpha_col_name else alpha)
        else:
            ax.scatter(df[x], df[y],
                       s=marker_sizes if size_col_name else marker_size,
                       c=df[response_column_name],
                       cmap=heatmap_palette,
                       edgecolor=marker_edge_color,
                       marker=marker,
                       linewidth=marker_edge_width,
                       alpha=alpha_values if alpha_col_name else alpha)
    
   # general plotting formatting
    ax.set_xlabel(xlabel if xlabel else x, fontsize=fontsize)
    ax.set_ylabel(ylabel if ylabel else y, fontsize=fontsize)
    if threedim:
        ax.set_zlabel(zlabel if zlabel else z, fontsize=fontsize)
    
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    if not threedim:
        ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    
    # make less white space
    if not threedim:
        # make less white space 
        fig.subplots_adjust(
        left=0.20,   # space between left edge of figure and first subplot
        right=.99,  # space between right edge of figure and last subplot
        bottom=0.17, # space between bottom edge and subplots
        top=0.98,    # space between top edge and subplots
        wspace=0.25, # horizontal space between subplots
        hspace=0.25)  # vertical space between subplots (not used here since only 1 row)

    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved as {plot_filename}")

    return fig, ax

def get_hopkins_stat(features):
  d = features.shape[1]
  n = len(features) # rows
  m = int(0.1 * n)
  
  nbrs = NearestNeighbors(n_neighbors=1).fit(features.values)
  rand_features = sample(range(0, n, 1), m)
 
  ujd = []
  wjd = []
  for j in range(0, m):
     u_dist, _ = nbrs.kneighbors(uniform(np.amin(features,axis=0),np.amax(features,axis=0),d).reshape(1, -1), 2, return_distance=True)
     ujd.append(u_dist[0][1])
     w_dist, _ = nbrs.kneighbors(features.iloc[rand_features[j]].values.reshape(1, -1), 2, return_distance=True)
     wjd.append(w_dist[0][1])
 
  hopkins_stat = sum(ujd) / (sum(ujd) + sum(wjd))
  if isnan(hopkins_stat):
     print(ujd, wjd)
     hopkins_stat = 0
 
  return hopkins_stat
  
#### ---- clustering evaluation functions ---- ####
def silhouette_scores(data, X_coordinate_column, Y_coordinate_column,
                      min_clusters, max_clusters, clustering_function, random_state):
    scores = []
    clusters_range = range(min_clusters, max_clusters + 1)

    for n_clusters in clusters_range:
        model, _ = clustering_function(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters)  
        coordinates = data[[X_coordinate_column, Y_coordinate_column]]
        clusters = model.predict(coordinates)

        silhouette_avg = silhouette_score(coordinates, clusters)

        fig, ax = plt.subplots(figsize=(6, 6))
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=ax)
        visualizer.fit(coordinates)

        plt.xlabel('Silhouette Score')
        plt.title(f'Silhouette Score for {n_clusters} Clusters: {silhouette_avg:.2f}')
        plt.show()

        if len(set(clusters)) > 1:
            scores.append(silhouette_avg)
        else:
            scores.append(-1)

    plt.plot(clusters_range, scores, marker="o", linestyle="-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    plt.show()


def aic_bic_inertia_scores(data, X_coordinate_column, Y_coordinate_column,
                            min_clusters, max_clusters, clustering_function, random_state):
    n_clusters = np.arange(min_clusters, max_clusters + 1)
    aic_scores = []
    bic_scores = []
    inertia_scores = []

    coordinates = data[[X_coordinate_column, Y_coordinate_column]]

    for n in n_clusters:
        model, _ = clustering_function(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters=n)
        if hasattr(model, 'bic'):
            bic_scores.append(model.bic(coordinates)) 
        else:
            bic_scores.append(np.nan) 

        if hasattr(model, 'aic'):
            aic_scores.append(model.aic(coordinates))
        else:
            aic_scores.append(np.nan) 
        if hasattr(model, 'inertia_'):
            inertia_scores.append(model.inertia_)  
        else:
            inertia_scores.append(np.nan) 

    plt.figure(figsize=(6, 4))
    if any(np.isfinite(aic_scores)):
        plt.plot(n_clusters, aic_scores, marker="o", linestyle="-", label='AIC', color='blue')
    if any(np.isfinite(bic_scores)):
        plt.plot(n_clusters, bic_scores, marker="o", linestyle="-", label='BIC', color='green')
    if any(np.isfinite(inertia_scores)):
        plt.plot(n_clusters, inertia_scores, marker="o", linestyle="-", label='Inertia', color='red')

    plt.xlabel('Number of Components/Clusters')
    plt.ylabel('Score')
    plt.title(f'Clustering Model Evaluation ({clustering_function.__name__})')
    plt.legend(loc='best')
    plt.show()
    return {'aic': aic_scores, 'bic': bic_scores, 'inertia': inertia_scores}

    
#### ---- clustering functions ---- ####
def cluster_plots( df, x, y, cluster_column, xlabel, ylabel,
                    cluster_color_map='plasma', marker_size=50, marker_color='black', alpha=0.7,
                    marker='o', marker_edge_color='black', title=None, save_plot=False, plot_filename=None,
                    fontsize=12, title_fontsize=14, subgroup=False, subgroup_column_name=None,
                    subgroup_palette='tab20', size_col_name=None, fig_size=(6,6),
                    subgroup_plot_dict=None, subgroup_marker='s'):
    
    fig, ax = plt.subplots(figsize=fig_size)
    clusters = sorted(df[cluster_column].unique())

    if not subgroup:
        if cluster_color_map is None:
            cluster_color_map = plt.get_cmap('plasma')
        elif isinstance(cluster_color_map, str):
            cluster_color_map = plt.get_cmap(cluster_color_map)

        norm = mcolors.Normalize(vmin=min(clusters), vmax=max(clusters))

        ax.scatter(df[x], df[y],
                    s=marker_size, marker=marker,
                    c=df[cluster_column],
                    cmap=cluster_color_map, alpha=alpha,
                    edgecolor=marker_edge_color, linewidth=1)
        
        handles = []
        for cluster in clusters:
            color = cluster_color_map(norm(cluster))
            handles.append(
                plt.Line2D([], [], color=color, marker=marker,
                        linestyle='', markersize=8, label=f'cluster {cluster}')
            )

        ax.legend(handles=handles, title="clusters", fontsize=fontsize, title_fontsize=fontsize,
                  bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    #if you want clusters + your subgroup plotted together
    else:
        if subgroup_column_name is None:
            raise ValueError("subgroup_column_name must be provided when subgroup is True")
        if subgroup_column_name not in df.columns:
            subgroup_column = df[subgroup_column_name]
            df[subgroup_column_name] = subgroup_column

        if cluster_color_map is None:
            cluster_color_map = plt.get_cmap('plasma')
        elif isinstance(cluster_color_map, str):
            cluster_color_map = plt.get_cmap(cluster_color_map)

        plt.scatter(df[x], df[y],
                    s=marker_size, marker=marker,
                    c=df[cluster_column], cmap=cluster_color_map, alpha=alpha,
                    edgecolor=marker_edge_color, linewidth=1, zorder=1)
        if subgroup_palette is None:
            subgroup_palette = plt.get_cmap('tab20')
        elif isinstance(subgroup_palette, str):
            subgroup_palette = plt.get_cmap(subgroup_palette)
        subgroup_labels = [label for label, action in subgroup_plot_dict.items() if action == 'plot']
        for i, type_label in enumerate(subgroup_labels):
            subset = df[df[subgroup_column_name] == type_label]
            color = subgroup_palette(i % 20)  # cycle through subgroup palette
            ax.scatter(
                subset[x], subset[y],
                s=marker_size , marker=subgroup_marker,
                c=[color], alpha=1.0,
                edgecolor=None, linewidth=1,
                label=str(type_label), zorder=2
            )
        handles = []

        # --- Cluster handles ---
        norm = mcolors.Normalize(vmin=min(clusters), vmax=max(clusters))
        for cluster in clusters:
            color = cluster_color_map(norm(cluster))
            handles.append(
                plt.Line2D([], [], color=color, marker=marker, linestyle='',
                        markersize=8, label=f'cluster {cluster}')
            )

        # --- Subgroup handles ---
        subgroup_labels = [label for label, action in subgroup_plot_dict.items() if action == 'plot']
        for i, type_label in enumerate(subgroup_labels):
            color = subgroup_palette(i % 20)  # cycle through subgroup palette
            handles.append(
                plt.Line2D([], [], color=color, marker=subgroup_marker, linestyle='',
                        markersize=8, label=str(type_label))
            )

        ax.legend(handles=handles, title="Clusters & Subgroups",
                fontsize=fontsize, title_fontsize=fontsize,
                bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # general plotting formatting
    ax.set_xlabel(xlabel if xlabel else x, fontsize=fontsize)
    ax.set_ylabel(ylabel if ylabel else y, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    
    # make less white space
    fig.subplots_adjust(
    left=0.20,   # space between left edge of figure and first subplot
    right=.99,  # space between right edge of figure and last subplot
    bottom=0.17, # space between bottom edge and subplots
    top=0.98,    # space between top edge and subplots
    wspace=0.25, # horizontal space between subplots
    hspace=0.25)  # vertical space between subplots (not used here since only 1 row)

    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved as {plot_filename}")
    return fig, ax

def kmeans_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(coordinates)  # Get cluster assignments
    data['cluster'] = clusters  # Store cluster labels in DataFrame

    return kmeans, data  # Return both the model and updated DataFrame

def gmm_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, covariance_type='full'):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type=covariance_type)
    gmm.fit(coordinates)
    clusters = gmm.predict(coordinates)
    data['cluster'] = clusters

    soft_labels = gmm.predict_proba(coordinates) #probability of a point belonging to each cluster
    soft_labels_df = pd.DataFrame(soft_labels, columns=[f'prob_belonging_to_cluster_{i}' for i in range(n_clusters)])
    
    data_w_probability= pd.concat([data, soft_labels_df], axis=1)

    return gmm, data_w_probability  # Return both the model and updated DataFrame

def hdbscan_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,min_cluster_size, min_samples, distance_metric):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    hdbscan_clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=distance_metric)
    hdbscan_clusters = hdbscan_clustering.fit_predict(coordinates)  
    clustered_points = (hdbscan_clusters != -1)  # points that are not noise
    
    data['cluster'] = hdbscan_clusters 
    print("Fraction of points clustered:",np.sum(clustered_points) / data.shape[0])
    
    return hdbscan_clustering, data 


def dbscan_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,  distance_metric, eps=0.5, min_samples=5):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    dbscan_clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric)
    dbscan_clusters = dbscan_clustering.fit_predict(coordinates)  
    clustered_points = (dbscan_clusters != -1)  # points that are not noise
    data['cluster'] = dbscan_clusters 
    
    print("Fraction of points clustered:",np.sum(clustered_points) / data.shape[0])
    
    return dbscan_clustering, data 


def agglomerative_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, distance_metric='euclidean', linkage_type='ward'):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    agg_clustering= AgglomerativeClustering(n_clusters=n_clusters, metric=distance_metric, linkage=linkage_type)
    agg_clusters = agg_clustering.fit_predict(coordinates)
    data['cluster'] = agg_clusters

    return agg_clustering, data

def elbow_method(data, X_coordinate_column, Y_coordinate_column, min_clusters, max_clusters, clustering_function, random_state):
    inertia = []
    cluster_range = range(min_clusters, max_clusters + 1)
    for n in cluster_range:
        model, _ = clustering_function(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, plot=False)  
        coordinates = data[[X_coordinate_column, Y_coordinate_column]]
        clusters = model.fit_predict(coordinates)
        cluster_variance = sum(np.var(coordinates[clusters == i], axis=0).sum() for i in np.unique(clusters))
        inertia.append(cluster_variance)

    plt.plot(cluster_range, inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Variance')
    plt.title('Elbow Method for Choosing n')
    plt.show()

    return inertia
    
    
def BIRCH_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, 
                     n_clusters, threshold, branching_factor):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    birch_clustering= Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor )
    birch_clusters = birch_clustering.fit_predict(coordinates)
    data['cluster'] = birch_clusters

    return birch_clustering, data

