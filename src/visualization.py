from typing import Callable, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go



def plot_scatter(x, y, title):
    sns.scatterplot(x, y, alpha=0.4)
    plt.xlabel('Original')
    plt.ylabel('Reconstruction')
    plt.title(title, fontsize=15)

def plot_distplot(x, y, title):
    sns.distplot(x, kde=False, label='Original')
    sns.distplot(y, kde=False, label='Reconstructed')
    plt.legend()
    plt.title(title, fontsize=15)

def plot_single_hist(x, title):
    sns.distplot(x, kde=False)
    plt.xlabel('residual')
    plt.ylabel('amount')
    plt.title(title, fontsize=15)
    

def plot_grid(vis_func: Callable, preds: np.array, labels: np.array, feature_names: Iterable[str], title: str):
    """Plots 2x2 grid of different plots.
    params:
        vis_func: function - takes args (x, y, title) and plots 1 plot for 1 cell
        preds: np.array with size (N, 4)
        labels: np.array with size (N, 4)
        feature_names: list with len==4 - contains names of features for visualisation
        title: str - main title of plot
    """
    fig, axes = plt.subplots(figsize=(12, 12))

    plt.subplot(221)
    vis_func(labels[:, 0], preds[:, 0], feature_names[0])
    
    plt.subplot(222)
    vis_func(labels[:, 1], preds[:, 1], feature_names[1])

    plt.subplot(223)
    vis_func(labels[:, 2], preds[:, 2], feature_names[2])

    plt.subplot(224)
    vis_func(labels[:, 3], preds[:, 3], feature_names[3])
    
    fig.suptitle(title, fontsize=16)
    

def plot_histograms(x, feature_names, title=''):
    """Plots 2x2 grid with histograms
    x: np.array with size (N, 4)
    feature_names: list with len==4 - contains names of features for visualisation
    title: str - main title of plot
    """
    fig, axes = plt.subplots(figsize=(12, 12))

    plt.subplot(221)
    plot_single_hist(x[:, 0], feature_names[0])

    plt.subplot(222)
    plot_single_hist(x[:, 1], feature_names[1])

    plt.subplot(223)
    plot_single_hist(x[:, 2], feature_names[2])

    plt.subplot(224)
    plot_single_hist(x[:, 3], feature_names[3])

    fig.suptitle(title, fontsize=16)
    
    
def plot_pairplot(data: pd.DataFrame, title: str):
    """Plots low-triangular matrix with scatter plots of features
    at non-diagonal cells and kde plots at diagonal cells.
    params:
        data: pd.DataFrame - df with data to visualise
        title: str - Main title of plot
    """
    sns.pairplot(
        data, corner=True,
        diag_kind="kde", markers="x", kind='reg',
        plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}},
    )
    plt.suptitle(title, fontsize=16)
    plt.plot()
    

def plot_3d_scatter(df, x: str, y: str, z: str, color: str):
    """Plots 3d interactive scatter plot with colored samples
    df: pd.DataFrame
    x: str - name of column in df to be in x axis
    y: str - name of column in df to be in y axis
    z: str - name of column in df to be in z axis
    color: str - name of column in df to set color of samples
    """
    
    print(f'Color is {color}')
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x],
        y=df[y],
        z=df[z],
        mode='markers',
        marker=dict(
            size=5,
            color=df[color],
            colorscale='Viridis',
            opacity=0.8,
            showscale=True,
        )
    )])
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="#7f7f7f"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene = dict(
            xaxis = dict(title=x),
            yaxis = dict(title=y),
            zaxis = dict(title=z)
        )
    )
    fig.show()