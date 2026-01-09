import os
import tqdm
import shutil
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tools.dirtools as nav
from IPython.display import Image, display

# ========== Simple functions ========== #

def common_col_title(fig, titles, shape):
    """Put a common `title` on the columns of figure `fig`.
    
    Args:
        - fig (plt.figure)
        - titles (list): list of strings, must have length = N2
        - shape (tuple): shape of figure subplots, (N1, N2)
    """
    N1, N2 = shape
    for n in range(N2):
        ax = fig.add_subplot(N1, N2, n+1, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax.set_title(titles[n])
        
def common_col_xlabel(fig, xlabels, shape):
    """Put a common `xlabel` on the columns of figure `fig`.
    
    Args:
        - fig (plt.figure)
        - titles (list): list of strings, must have length = N2
        - shape (tuple): shape of figure subplots, (N1, N2)
    """
    N1, N2 = shape
    for n in range(N2):
        ax = fig.add_subplot(N1, N2, (N1-1)*N2 + n+1, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax.set_xlabel(xlabels[n])

def common_row_ylabel(fig, ylabels, shape):
    """Put a common `ylabel` on the rows of figure `fig`.
    
    Args:
        - fig (plt.figure)
        - titles (list): list of strings, must have length = N1
        - shape (tuple): shape of figure subplots, (N1, N2)
    """
    N1, N2 = shape
    for n in range(N1):
        ax = fig.add_subplot(N1, N2, n * N2 + 1, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax.set_ylabel(ylabels[n])

def common_label(fig, xlabel, ylabel):
    """Put a common `xlabel` and `ylabel` on the figure `fig`.
    
    Args:
        - fig (plt.figure)
        - xlabel (str)
        - ylabel (str)
    """
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def savefig(figname="temp.png", clear=True, close=False, dpi=200, folders=[]):
    """Saves figure.
    
    Args:
        - figname (str): default: "temp.png"
        - clear (bool): whether to execute plt.clf(), default: True
        - close (bool): whether to close all plots, default: False
        - dpi (int): default: 200
        - folders (list): parent folders, default: []
    """
    if len(folders) > 0: nav.mkfile(os.path.join(*folders))
    plt.tight_layout()
    plt.savefig(os.path.join(*folders, figname), dpi=dpi)
    if clear: plt.clf()
    if close: plt.close("all")
    
def color_time(ax, x, cmap="viridis"):
    """Colors the (`x`, `y`, *`z`) trajectory by time on the axis `ax`.
    
    Args:
        - ax (plt.subplot): axis object to plot on
        - x (np.array): trajectory to plot, shape = (dim, time), where dim \in {2,3}
    """
    T = x.shape[1] - 2
    color = sns.color_palette(cmap, T)
    for t in range(T):
        ax.plot(*x[:, t:t+2], color=color[t], alpha=0.5, marker="")
        
def set_xylabel(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
        
def set_invisible(ax, rm_all=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if rm_all:
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
def rm_ticklabels(ax, rm_ticks=True, rm_labels=True):
    if rm_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if rm_labels:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        
