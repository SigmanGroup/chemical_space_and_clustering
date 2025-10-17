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

# --- Main scatterplot function ---
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




# making plots prettier
def map_alpha_by_col(df, alpha_val_map=None, type_col=None, alpha=0.5):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if alpha_val_map is None:
        alpha_val_map = {}
    
    df['alpha'] = df[type_col].map(alpha_val_map).fillna(alpha)
    return df

def map_color_by_col(df, color_map=None, type_col=None, color='hotpink'):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if color_map is None:
        color_map = {}
    
    df['color'] = df[type_col].map(color_map).fillna(color)
    return df
    
def map_size_by_col(df, size_map=None, type_col=None, size=100):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if size_map is None:
        size_map = {}
    
    df['size'] = df[type_col].map(size_map).fillna(size)
    return df
    
def map_marker_by_col(df, marker_map=None, type_col=None, marker='o'):
    if type_col is None:
        type_col_avail = [col for col in ['type', 'class', 'label', 'labels'] if col in df.columns]
        if len(type_col_avail) > 1:
            raise ValueError("Multiple options for type column found, specify column using type_col = 'column_name'")
        elif len(type_col_avail) == 1:
            type_col = type_col_avail[0]
        elif type_col == None:
            raise ValueError("No type column found, specify column using type_col = 'column_name'")

    if marker_map is None:
        marker_map = {}
    
    df['marker'] = df[type_col].map(marker_map).fillna(marker)
    return df


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
  
  
def plot_umap(features, ID, image, smiles, n_neighbors, min_dist, distance_metric, random_state, 
              font_size=16, type_column=None, response_column=None, alpha=0.7, color='#7B1B79', marker='o', size=100,  
              alpha_value_mapping=None, color_mapping=None, marker_mapping=None, size_mapping=None, plot_order=None,
              export_excel=False, excel_filename="umap_coordinates_df.xlsx",
              save_plot=False, plot_filename="umap_plot.png", heatmap=False, palette='plasma'):

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2,
                        metric=distance_metric, random_state=random_state)
    
    umap_results = reducer.fit_transform(features)
    umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
    umap_df['name'], umap_df['image'], umap_df['smiles'] = ID, image, smiles

    plt.figure(figsize=(6, 6))

    if heatmap:
        umap_df['response'] = response_column
        norm = plt.Normalize(umap_df['response'].min(), umap_df['response'].max())
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=alpha, c=umap_df['response'], cmap=palette, marker=marker, s=size)
        plt.colorbar(sm, label='Response')

    elif type_column is not None:
        umap_df['type'] = type_column
        umap_df['type'] = umap_df['type'].astype(str)
        unique_types = sorted(umap_df['type'].unique())

        cmap = cm.get_cmap('tab20', len(unique_types))
        default_colors = [cmap(i) for i in range(len(unique_types))]
        default_markers = itertools.cycle(['s', 'o', 'D', '^', 'v', '*', 'X', 'P'])
        default_sizes = itertools.cycle([50, 80, 100, 120, 140, 160])

        color_mapping = color_mapping or {t: default_colors[i] for i, t in enumerate(unique_types)}
        marker_mapping = marker_mapping or {t: next(default_markers) for t in unique_types}
        size_mapping = size_mapping or {t: next(default_sizes) for t in unique_types}
        alpha_value_mapping = alpha_value_mapping or {t: 0.8 for t in unique_types}

        for t in unique_types:
            t = str(t) #corrects error if column contains true/false and is read as a boolean 
            umap_df['type'] = umap_df['type'].astype(str)
            
            subset = umap_df[umap_df['type'] == t]
            plt.scatter(subset['UMAP1'], subset['UMAP2'], 
                        alpha=alpha_value_mapping[t], color=color_mapping[t], 
                        marker=marker_mapping[t], s=size_mapping[t], label=t)
        
        plt.legend(title='Type', fontsize=12)

    else:
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=alpha, color=color, marker=marker, s=size)

    plt.xlabel('UMAP1', fontsize=font_size)
    plt.ylabel('UMAP2', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.grid(False)

    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")

    if export_excel:
        umap_df.to_excel(excel_filename, index=False)
        print(f"UMAP data saved to {excel_filename}")

    plt.show()
    return umap_df

def plot_tsne(features, ID, image, smiles, perplexity, early_exaggeration,learning_rate, distance_metric, random_state, init='random',
              font_size=16, type_column=None, response_column=None, alpha=0.7, color='#7B1B79', marker='o', size=100,  
              alpha_value_mapping=None, color_mapping=None, marker_mapping=None, size_mapping=None, plot_order=None,
            export_excel=False, excel_filename="tsne_coordinates_df.xlsx",
            save_plot=False, plot_filename="tsne_plot.png", heatmap=False, palette='plasma'):
              
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, metric=distance_metric, 
            init=init, random_state=random_state, )
    tsne_results = tsne.fit_transform(features)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['name'], tsne_df['image'], tsne_df['smiles'] = ID, image, smiles
    plt.figure(figsize=(6, 6))

    if heatmap:
        tsne_df['response'] = response_column
        norm = plt.Normalize(tsne_df['response'].min(), tsne_df['response'].max())
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], alpha=alpha, c=tsne_df['response'], cmap=palette, marker=marker, s=size)
        plt.colorbar(sm, label='Response')

    elif type_column is not None:
        tsne_df['type'] = type_column
        unique_types = sorted(tsne_df['type'].unique())

        cmap = cm.get_cmap(palette, len(unique_types))
        default_colors = [cmap(i) for i in range(len(unique_types))]
        default_markers = itertools.cycle(['s', 'o', 'D', '^', 'v', '*', 'X', 'P'])
        default_sizes = itertools.cycle([50, 80, 100, 120, 140, 160])

        color_mapping = color_mapping or {t: default_colors[i] for i, t in enumerate(unique_types)}
        marker_mapping = marker_mapping or {t: next(default_markers) for t in unique_types}
        size_mapping = size_mapping or {t: next(default_sizes) for t in unique_types}
        alpha_value_mapping = alpha_value_mapping or {t: 0.8 for t in unique_types}

        for t in unique_types:
            subset = tsne_df[tsne_df['type'] == t]
            plt.scatter(subset['TSNE1'], subset['TSNE2'], 
                        alpha=alpha_value_mapping[t], color=color_mapping[t], 
                        marker=marker_mapping[t], s=size_mapping[t], label=t)
        
        plt.legend(title='Type', fontsize=12)

    else:
        plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], alpha=alpha, color=color, marker=marker, s=size)

    plt.xlabel('TSNE1', fontsize=font_size)
    plt.ylabel('TSNE2', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.grid(False)

    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")

    if export_excel:
        tsne_df.to_excel(excel_filename, index=False)
        print(f"TSNE data saved to {excel_filename}")

    plt.show()
    return tsne_df
    
    
def plot_pca(feature_df, features, ID, image, smiles, n_components=2, alpha=0.7, color='#7B1B79', marker='o', size=100,  
             alpha_value_mapping=None, color_mapping=None, marker_mapping=None, size_mapping=None, plot_order=None,
             pc_x='PC1', pc_y='PC2', pc_z=None, font_size=16, type_column=None, response_column=None,
             export_excel=False, export_feature_excel=False, excel_filename="pca_coordinates_df.xlsx",
             save_plot=False, plot_filename="pca_plot.png", feature_filename = "pca_feature_loading.xlsx", heatmap=False, palette='plasma'):
    
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(features)
    pca_score = pca.explained_variance_ratio_
    
    print('Total variance explained by PCs:', round(np.sum(pca_score * 100), 1), '%\n')
    print("Percentage of explained variance per principal component:")
    for i, j in enumerate(pca_score):
        print(f"PC{i+1}   {j * 100:.1f}%")
    
    pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['name'], pca_df['image'], pca_df['smiles'] = ID, image, smiles
    
    fig = plt.figure(figsize=(6, 6))
    
    if pc_z is None:
        ax = plt.gca()  # 2D plot
        if heatmap:
            pca_df['response'] = response_column
            norm = plt.Normalize(pca_df['response'].min(), pca_df['response'].max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            plt.scatter(pca_df[pc_x], pca_df[pc_y], alpha=alpha, c=pca_df['response'], cmap=palette, marker=marker, s=size)
            plt.colorbar(sm, label='Response')
        elif type_column is not None:
            pca_df['type'] = type_column
            unique_types = sorted(pca_df['type'].unique())
            cmap = cm.get_cmap('tab20', len(unique_types))
            default_colors = [cmap(i) for i in range(len(unique_types))]
            default_markers = itertools.cycle(['s', 'o', 'D', '^', 'v', '*', 'X', 'P'])
            default_sizes = itertools.cycle([50, 80, 100, 120, 140, 160])
            color_mapping = color_mapping or {t: default_colors[i] for i, t in enumerate(unique_types)}
            marker_mapping = marker_mapping or {t: next(default_markers) for t in unique_types}
            size_mapping = size_mapping or {t: next(default_sizes) for t in unique_types}
            alpha_value_mapping = alpha_value_mapping or {t: 0.8 for t in unique_types}
            for t in unique_types:
                subset = pca_df[pca_df['type'] == t]
                plt.scatter(subset[pc_x], subset[pc_y], 
                            alpha=alpha_value_mapping[t], color=color_mapping[t], 
                            marker=marker_mapping[t], s=size_mapping[t], label=t)
            plt.legend(title='Type', fontsize=12)
        else:
            plt.scatter(pca_df[pc_x], pca_df[pc_y], alpha=alpha, color=color, marker=marker, s=size)
        plt.xlabel(pc_x, fontsize=font_size)
        plt.ylabel(pc_y, fontsize=font_size)
    
    else:
        ax = fig.add_subplot(111, projection='3d')  # 3D plot
        if type_column is not None:
            for t in unique_types:
                subset = pca_df[pca_df['type'] == t]
                ax.scatter(subset[pc_x], subset[pc_y], subset[pc_z], 
                           alpha=alpha_value_mapping[t], color=color_mapping[t], 
                           marker=marker_mapping[t], s=size_mapping[t], label=t)
            plt.legend(title='Type', fontsize=12)
        else:
            ax.scatter(pca_df[pc_x], pca_df[pc_y], pca_df[pc_z], alpha=alpha, color=color, marker=marker, s=size)
        ax.set_xlabel(pc_x, fontsize=font_size)
        ax.set_ylabel(pc_y, fontsize=font_size)
        ax.set_zlabel(pc_z, fontsize=font_size)
    
    ax.tick_params(labelsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.grid(False)
    
    feature_names = feature_df.columns if isinstance(feature_df, pd.DataFrame) else [f'Feature {i+1}' for i in range(feature_df.shape[1])]
    feature_contributions = pd.DataFrame(pca.components_.T, 
                                         index=feature_names, 
                                         columns=[f'PC{i+1}' for i in range(n_components)])
                                         

    if export_feature_excel:
        feature_contributions.to_excel(feature_filename)
        print(f"Feature contributions saved to {feature_filename}")
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
    
    if export_excel:
        pca_df.to_excel(excel_filename, index=False)
        print(f"PCA data saved to {excel_filename}")
    
    plt.show()
    return pca_df

def plot_feature_v_feature(features, ID, image, smiles, feature_x, feature_y, 
                           feature_z=None, font_size=16, type_column=None, response_column=None,
                           alpha=0.7, color='#7B1B79', marker='o', size=100,  
                           alpha_value_mapping=None, color_mapping=None, 
                           marker_mapping=None, size_mapping=None, plot_order=None,
                           export_excel=False, excel_filename="feature1_vs_feature2_coordinates_df.xlsx",
                           save_plot=False, plot_filename="feature1_vs_feature2_plot.png",
                           heatmap=False, palette='plasma'):
    
    chem_space_df = features[[feature_x, feature_y] + ([feature_z] if feature_z else [])].copy()
    chem_space_df['name'], chem_space_df['image'], chem_space_df['smiles'] = ID, image, smiles
    fig = plt.figure(figsize=(6, 6))

    if feature_z is None:
        ax = plt.gca()
        if heatmap:
            chem_space_df['response'] = response_column
            norm = plt.Normalize(chem_space_df['response'].min(), chem_space_df['response'].max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            plt.scatter(chem_space_df[feature_x], chem_space_df[feature_y], alpha=alpha, 
                        c=chem_space_df['response'], cmap=palette, marker=marker, s=size)
            plt.colorbar(sm, label='Response')

        elif type_column is not None:
            chem_space_df['type'] = type_column
            unique_types = sorted(chem_space_df['type'].unique())

            cmap = cm.get_cmap('tab20', len(unique_types))
            color_mapping = {t: cmap(i) for i, t in enumerate(unique_types)}
            marker_mapping = {t: m for t, m in zip(unique_types, itertools.cycle(['s', 'o', 'D', '^', 'v', '*', 'X', 'P']))}
            size_mapping = {t: s for t, s in zip(unique_types, itertools.cycle([50, 80, 100, 120, 140, 160]))}
            alpha_value_mapping = {t: 0.8 for t in unique_types}

            for t in unique_types:
                subset = chem_space_df[chem_space_df['type'] == t]
                plt.scatter(subset[feature_x], subset[feature_y], 
                            alpha=alpha_value_mapping[t], color=color_mapping[t], 
                            marker=marker_mapping[t], s=size_mapping[t], label=t)
            
            plt.legend(title='Type', fontsize=12)

        else:
            plt.scatter(chem_space_df[feature_x], chem_space_df[feature_y], alpha=alpha, 
                        color=color, marker=marker, s=size)
        
        plt.xlabel(feature_x, fontsize=font_size)
        plt.ylabel(feature_y, fontsize=font_size)
    else:
        ax = fig.add_subplot(111, projection='3d')
        if heatmap:
            chem_space_df['response'] = response_column
            norm = plt.Normalize(chem_space_df['response'].min(), chem_space_df['response'].max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            ax.scatter(chem_space_df[feature_x], chem_space_df[feature_y], chem_space_df[feature_z], 
                       alpha=alpha, c=chem_space_df['response'], cmap=palette, marker=marker, s=size)
            fig.colorbar(sm, ax=ax, label='Response')

        elif type_column is not None:
            chem_space_df['type'] = type_column
            unique_types = sorted(chem_space_df['type'].unique())

            cmap = cm.get_cmap('tab20', len(unique_types))
            color_mapping = {t: cmap(i) for i, t in enumerate(unique_types)}
            marker_mapping = {t: m for t, m in zip(unique_types, itertools.cycle(['s', 'o', 'D', '^', 'v', '*', 'X', 'P']))}
            size_mapping = {t: s for t, s in zip(unique_types, itertools.cycle([50, 80, 100, 120, 140, 160]))}
            alpha_value_mapping = {t: 0.8 for t in unique_types}

            for t in unique_types:
                subset = chem_space_df[chem_space_df['type'] == t]
                ax.scatter(subset[feature_x], subset[feature_y], subset[feature_z], 
                           alpha=alpha_value_mapping[t], color=color_mapping[t], 
                           marker=marker_mapping[t], s=size_mapping[t], label=t)
            
            plt.legend(title='Type', fontsize=12)

        else:
            ax.scatter(chem_space_df[feature_x], chem_space_df[feature_y], chem_space_df[feature_z], 
                       alpha=alpha, color=color, marker=marker, s=size)

        ax.set_xlabel(feature_x, fontsize=font_size)
        ax.set_ylabel(feature_y, fontsize=font_size)
        ax.set_zlabel(feature_z, fontsize=font_size)

    ax.tick_params(labelsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.grid(False)

    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")

    if export_excel:
        chem_space_df.to_excel(excel_filename, index=False)
        print(f"Feature data saved to {excel_filename}")

    plt.show()
    return chem_space_df

def silhouette_scores(data, X_coordinate_column, Y_coordinate_column,
                      min_clusters, max_clusters, clustering_function, random_state):
    scores = []
    clusters_range = range(min_clusters, max_clusters + 1)

    for n_clusters in clusters_range:
        model, _ = clustering_function(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, plot=False)  
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
    
    
def kmeans_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, plot=True):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(coordinates)  # Get cluster assignments
    data['cluster'] = clusters  # Store cluster labels in DataFrame
    
    if plot:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X_coordinate_column, y=Y_coordinate_column, hue='cluster', 
                        data=data, palette='viridis', s=100)
        plt.xlabel(X_coordinate_column)
        plt.ylabel(Y_coordinate_column)
        plt.legend(title='Cluster')
        plt.show()

    return kmeans, data  # Return both the model and updated DataFrame


def aic_bic_inertia_scores(data, X_coordinate_column, Y_coordinate_column,
                            min_clusters, max_clusters, clustering_function, random_state):
    n_clusters = np.arange(min_clusters, max_clusters + 1)
    aic_scores = []
    bic_scores = []
    inertia_scores = []

    coordinates = data[[X_coordinate_column, Y_coordinate_column]]

    for n in n_clusters:
        model, _ = clustering_function(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters=n, plot=False)
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

def gmm_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, covariance_type='full', plot=True):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type=covariance_type)
    gmm.fit(coordinates)
    clusters = gmm.predict(coordinates)
    data['cluster'] = clusters

    soft_labels = gmm.predict_proba(coordinates) #probability of a point belonging to each cluster
    soft_labels_df = pd.DataFrame(soft_labels, columns=[f'prob_belonging_to_cluster_{i}' for i in range(n_clusters)])
    
    data_w_probability= pd.concat([data, soft_labels_df], axis=1)
    
    if plot:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X_coordinate_column, y=Y_coordinate_column, hue='cluster', 
                        data=data, palette='viridis', s=100)
        plt.xlabel(X_coordinate_column)
        plt.ylabel(Y_coordinate_column)
        plt.legend(title='Cluster')
        plt.show()

    return gmm, data_w_probability  # Return both the model and updated DataFrame

def hdbscan_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,min_cluster_size, min_samples, distance_metric, plot=True):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    hdbscan_clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=distance_metric)
    hdbscan_clusters = hdbscan_clustering.fit_predict(coordinates)  
    clustered_points = (hdbscan_clusters != -1)  # points that are not noise
    
    data['cluster'] = hdbscan_clusters 
    print("Fraction of points clustered:",np.sum(clustered_points) / data.shape[0])
    
    if plot:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X_coordinate_column, y=Y_coordinate_column, hue='cluster', 
                        data=data, palette='viridis', s=100, alpha=0.7)
        plt.xlabel(X_coordinate_column)
        plt.ylabel(Y_coordinate_column)
        plt.legend(title='Cluster')
        plt.show()

    return hdbscan_clustering, data 

def dbscan_clustering(data, X_coordinate_column, Y_coordinate_column, random_state,  distance_metric, eps=0.5, min_samples=5, plot=True):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    dbscan_clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric)
    dbscan_clusters = dbscan_clustering.fit_predict(coordinates)  
    clustered_points = (dbscan_clusters != -1)  # points that are not noise
    data['cluster'] = dbscan_clusters 
    
    print("Fraction of points clustered:",np.sum(clustered_points) / data.shape[0])
    
    if plot:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X_coordinate_column, y=Y_coordinate_column, hue='cluster', 
                        data=data, palette='viridis', s=100)
        plt.xlabel(X_coordinate_column)
        plt.ylabel(Y_coordinate_column)
        plt.legend(title='Cluster')
        plt.show()

    return dbscan_clustering, data 


def agglomerative_clustering(data, X_coordinate_column, Y_coordinate_column, random_state, n_clusters, distance_metric='euclidean', linkage_type='ward', plot=True):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    agg_clustering= AgglomerativeClustering(n_clusters=n_clusters, metric=distance_metric, linkage=linkage_type)
    agg_clusters = agg_clustering.fit_predict(coordinates)
    data['cluster'] = agg_clusters

    if plot:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X_coordinate_column, y=Y_coordinate_column, hue='cluster', 
                        data=data, palette='viridis', s=100)
        plt.xlabel(X_coordinate_column)
        plt.ylabel(Y_coordinate_column)
        plt.legend(title='Cluster')
        plt.show()

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
                     n_clusters, threshold, branching_factor, plot=True):
    coordinates = data[[X_coordinate_column, Y_coordinate_column]]
    birch_clustering= Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor )
    birch_clusters = birch_clustering.fit_predict(coordinates)
    data['cluster'] = birch_clusters

    if plot:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X_coordinate_column, y=Y_coordinate_column, hue='cluster', 
                        data=data, palette='viridis', s=100)
        plt.xlabel(X_coordinate_column)
        plt.ylabel(Y_coordinate_column)
        plt.legend(title='Cluster')
        plt.show()

    return birch_clustering, data


