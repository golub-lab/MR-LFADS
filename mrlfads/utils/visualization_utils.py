import os
import tqdm
import shutil
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image, display

import mrlfads.utils.dir_utils as nav

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
        
# ========== Plots ========== #

def fill_between(mean, std, color, alpha=0.3, ax=None, **kwargs):
    if ax == None: ax = plt.figure().add_subplot(111)
    ax.plot(mean, color=color, **kwargs)
    ax.fill_between(
        range(len(mean)),
        mean - std,
        mean + std,
        color=color,
        alpha=alpha
    )
    return ax

# ========== Gifs ========== #

def gen_gif_by_file(foldername, files, stall=5, rm=False, folders=[]):
    """Generate gifs from `files` in `foldername`, where each image is repeated `stall` times.
    
    Args:
        - foldername (str): name of the folder that contains the images
        - files (list): list of image file names
        - stall (int): number of times to repeat the image, default: 5
        - rm (bool): whether to remove folder containing images after gif creation, default: False
        - folders (list): parent folders, default: []
    """
    nav.mkfile(os.path.join(*folders, foldername))
    images = []
    for f in tqdm.tqdm(files):
        for _ in range(stall):
            images.append(imageio.imread(os.path.join(*folders, foldername, f"{f}.png")))
    imageio.mimsave(os.path.join(*folders, f"{foldername}.gif"), images, duration=stall)
    if rm: shutil.rmtree(os.path.join(*folders, foldername))
    
def gen_gif_by_axis(foldername, ax, stall=5, angle1=30, angles=None, rm=False, folders=[]):
    """Generate gifs that is the same ``ax`` rotated.
    
    Args:
        - foldername (str): name of the folder that contains the images
        - ax (plt.subplot): axis object to plot
        - stall (int): number of times to repeat the image, default: 5
        - angle1 (int): the tilt of the x-y plane, default: 30
        - angles (array-like): the rotation angle about the z axis, default: None (automatic assignment)
        - rm (bool): whether to remove folder containing images after gif creation, default: False
        - folders (list): parent folders, default: []
    """
    nav.mkfile(os.path.join(*folders, foldername))
    
    if angles == None: angles = np.linspace(-180, 180, 20)
    for angle in angles:
        ax.view_init(angle1, angle)
        savefig(figname=f"{foldername}/{round(angle,0)}.png", clear=False)

    images = []
    for angle in tqdm.tqdm(angles):
        for _ in range(stall):
            images.append(imageio.imread(os.path.join(*folders, foldername, f"{round(angle,0)}.png")))
    imageio.mimsave(os.path.join(*folders, f"{foldername}.gif"), images)
    if rm: shutil.rmtree(os.path.join(*folders, foldername))
    
def gen_gif_by_code(ts, foldername, folders=".", stall=5, ipynb_display=False, rm=False, *args, **kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for ti in ts:
                func(ti, *args, **kwargs)
                savefig(f"{ti}.png", folders=[folders, foldername], clear=True, close=True)
            files = [str(ti) for ti in ts]
            gen_gif_by_file(foldername, files, stall=stall, folders=[folders], rm=rm)
            
            if ipynb_display:
                gif_path = os.path.join(*folders, f"{foldername}.gif")
                display(Image(filename=gif_path))
        return wrapper
    return decorator
    
# ========== Trajectory-related ========== #
    
def plot_trajectory(trajectories, figname="temp.png", save=True, plot_time=True, ax=None, **kwargs):
    """Plot multiple `trajectories` in `ax`, either colored in time or not.
    
    Args:
        - trajectories (np.array): shape=(batch, dim, time), where dim \in {2,3}
        - ax (plt.subplot): axis object to plot on, default=None (generates automatically)
        - save (bool): whether to save the plot or not, default: True
        - figname (str): figure name, only used if save=True
        - plot_time (bool): whether to plot time or not, default: True
    Kwargs:
        - color (str): default="k"
        - xlabel, ylabel, subtitle (str)
    Returns:
        - ax (plt.subplot)
    """
    kw = {"color": "k", "xlabel": "x", "ylabel": "y", "subtitle": ""}
    kw.update(kwargs)

    batch, dim, time = trajectories.shape
    if ax == None: ax = plt.figure().add_subplot(111, projection=f"{dim}d")
    if plot_time:
        for b in range(batch): color_time(ax, trajectories[b])
    else:
        for b in range(batch): ax.plot(*trajectories[b], alpha=0.5, color=kw["color"])
    ax.set_xlabel(kw["xlabel"]); ax.set_ylabel(kw["ylabel"]); ax.set_title(kw["subtitle"])
    if save: savefig(figname)
    return ax

def plot_trajectory_gif(
    trajectories, foldername, angles=None, itvl=10, cmap="viridis", cmap_dict=None, folders=[], rotate=True,
):
    """Plot multiple `trajectories` in `ax`, colored, evolving and rotating in time.
    
    Args:
        - trajectories (np.array): shape = (batch, dim, time), where dim \in {2,3}
        - foldername (str): name of the folder that contains the images
        - angle1 (int): the tilt of the x-y plane, default: 30
        - angles (array-like): the rotation angle about the z axis, default: None (automatic assignment)
        - cmap (str): default: "viridis"
        - cmap_dict (dict): for batch-dependent coloring, {batch index: cmap}, default: None
        - folders (list): parent folders, default: []
    """
    # Setup folder, angles, cmap_dict
    nav.mkfile(*folders, foldername)
    batch, dim, time = trajectories.shape
    if angles == None: angles = np.linspace(-180, 180, 20)
    if not cmap_dict:
        cmap_dict = {}
        for b in range(batch): cmap_dict[b] = cmap
    
    # Plot per time step while rotating
    for t in range(0, time-1, itvl):
        ax = plt.figure().add_subplot(111, projection=f"{dim}d")
        if rotate: ax.view_init(30, angles[t % len(angles)])
        for b in range(batch):
            color_time(ax, trajectories[b][..., :t+1], cmap=cmap_dict[b])
        savefig(figname=f"{t}.png", clear=True, close=True, folders=[*folders, foldername])

    files = [f"{t}" for t in range(0, time-1, itvl)]
    gen_gif_by_file(foldername, files, stall=5, rm=False, folders=folders)

