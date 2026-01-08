"""
Callback groups for training and validation.

Main classes
------------
OnInitEndCalls:
    Run a sequence of callback objects once near the start of a run.

OnEpochEndCalls:
    Run callback objects at epoch end for training and/or validation.
"""

import io
import os
import torch
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from PIL import Image
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tools.vistools import common_label, common_col_title
from tools.dirtools import pklsave

plt.switch_backend("Agg")
SAVE_DIR = "./graphs"
    
# ========== Callback Stacks ========== #

class OnInitEndCalls(pl.Callback):
    """Callbacks executed once at initialization end.

    These callbacks are conceptually associated with the `on_init_end` phase
    but are invoked during `on_validation_epoch_start` in order to ensure
    access to the `trainer` and `pl_module` objects.
    """
    def __init__(self,
                 priority: int = 1,
                 callbacks: list = []
                ):
        self.priority = priority
        self.ran = False
        self.callbacks = callbacks
        
    def on_validation_epoch_start(self, trainer, pl_module):
        if self.ran: return
    
        # Extract model components, data, etc.
        dataloader = trainer.datamodule.val_dataloader()
        batches = next(iter(dataloader))
        batches, info_dict = zip(*batches.values())
        
        # Run individual callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        kwargs = {"batches": batches, "info": info_dict}
        for i, callback in enumerate(self.callbacks):
            callback.run(trainer, pl_module, **kwargs)

        # Run only once
        self.ran = True

class OnEpochEndCalls(pl.Callback):
    """Callbacks executed at the end of training or validation epochs.

    These callbacks are invoked during `on_train_epoch_end` or
    `on_validation_epoch_end`.
    """
    def __init__(self,
                 callbacks: list,
                 in_train: str,
                 priority: int = 1):
        self.priority = priority
        self.callbacks = callbacks
        self.sess_idx = 0
        
        self.in_train = in_train
        assert len(self.in_train) == len(callbacks)
        
    def on_train_epoch_end(self, trainer, pl_module):
        
        # Extract model components, data, etc.
        batch = pl_module.current_batch[self.sess_idx]
        save_var = pl_module.save_var
        kwargs = {"batch": batch, "save_var": save_var, "log_metrics": self.callbacks[0].metrics}
        
        # Run individual callbacks
        for i, callback in enumerate(self.callbacks):
            if int(self.in_train[i]):
                new_kwargs = callback.run(trainer, pl_module, **kwargs)
                if not isinstance(new_kwargs, type(None)): kwargs.update(new_kwargs)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        # Extract model components, data, etc.
        batches = pl_module.current_batch
        save_var = pl_module.save_var
        outputs = pl_module.outputs
        kwargs = {"batches": batches, "save_var": save_var, "outputs": outputs, "log_metrics": self.callbacks[0].metrics}
        
        # Run individual callbacks
        for i, callback in enumerate(self.callbacks):
            if not int(self.in_train[i]):
                new_kwargs = callback.run(trainer, pl_module, **kwargs)
                if not isinstance(new_kwargs, type(None)): kwargs.update(new_kwargs)
                
class Log:
    """Collect logged metrics from a PyTorch Lightning trainer.

    Notes:
        - This class assumes the TensorBoard logger is `trainer.loggers[0]`.
    """
    def __init__(self,
                 tags: list = []):
        self.name = "log"
        self.metrics = defaultdict(list)
        self.tags = tags
    
    def run(self, trainer, pl_module, **kwargs):
        new_metrics = trainer.logged_metrics
        self.update_dict(self.metrics, new_metrics)
        
        log_dir = trainer.loggers[0].log_dir # Tensorboard logger has to be 1st logger
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        scalers = event_acc.Tags()["scalars"]
        for tag in self.tags:
            if tag in scalers:
                scalar_events = event_acc.Scalars(tag)
                values = [event.value for event in scalar_events]
                self.metrics[tag] = values
            else:
                self.metrics[tag] = []
                
        return {"log_metrics": self.metrics}
        
    def update_dict(self, old_dict, new_dict):
        for key, value in new_dict.items():
            old_dict[key].append(value.item())

# ========== Reconstruction Plots ========== #
    
class InferredRatesPlot:
    """Plots inferred rates for heldin neurons with optionally smoothed data."""
    def __init__(
        self,
        n_samples=3,
        n_batches=4,
        log_every_n_epochs=10,
        smooth=True,                # plot smoothed version of data (e.g. for spikes)
        plot_first_session=True,    # only plot results for the first session
        plot_random_units=False,    # if False, plots units with the highest activities
    ):
        self.name = "inferred_rates_plot"
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.log_every_n_epochs = log_every_n_epochs
        
        if smooth:
            self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        else:
            self.smoothing_func = lambda x: x
            
        self.plot_first_session = plot_first_session
        self.plot_random_units = plot_random_units

    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
        
        # Get data and session
        batches, save_var, outputs = kwargs["batches"], kwargs["save_var"], kwargs["outputs"]
        if self.plot_first_session: sessions = [0]
        else: sessions = range(len(batches))
        
        for s in sessions:

            # Get data
            batch = batches[s]
            units = pl_module.maximum_activity_units(s, self.n_samples)
            ic_enc_seq_len = pl_module.hparams.ic_enc_seq_len

            # Create subplots
            n_rows, n_cols = len(pl_module.area_names) * self.n_samples, self.n_batches
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                sharex=True,
                sharey="row",
                figsize=(3 * n_cols, 2 * n_rows),
            )
            common_label(fig, "time step", "rates")
            common_col_title(fig, [f"Batch {i}" for i in range(n_cols)], (n_rows, n_cols))

            # Iterate through areas and take n_sample neurons
            count = 0
            for area_name, area in pl_module.areas.items():
                encod_data = batch.encod_data[area_name].detach().cpu().numpy()[:, ic_enc_seq_len:]
                
                # Process outputs based on distribution
                if area.output_dist.name == "poisson":
                    infer_data = torch.exp(outputs[area_name][s].detach().cpu()).numpy()
                else:
                    infer_data = outputs[area_name][s].detach().cpu().numpy()
                    
                # Choose units to plot
                if self.plot_random_units:
                    units_to_plot = np.random.choice(area.hparams.num_neurons[s], size=self.n_samples, replace=False)
                else: 
                    units_to_plot = units[area_name]

                for jn in units_to_plot:

                    for ib in range(self.n_batches):
                        # Plot spikes at negative locations
                        smoothed_value = self.smoothing_func(encod_data[ib, :, jn])
                        y_lim_bottom = - max(smoothed_value) / 2
                        spike_idx = np.nonzero(encod_data[ib, :, jn])
                        spike_loc = np.ones(len(spike_idx)) * y_lim_bottom
                        axes[count][ib].plot(spike_idx, spike_loc, color="darkgray", marker=".")
                        
                        # Plot inferred and smooth data
                        # axes[count][ib].plot(encod_data[ib, :, jn], "gray", alpha=0.5)
                        axes[count][ib].plot(infer_data[ib, :, jn], "b")
                        axes[count][ib].plot(smoothed_value, "k--")

                    axes[count][0].set_ylabel(f"area {area_name}, neuron #{jn}")
                    count += 1

            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/inferred_rates_plot_epoch{trainer.current_epoch}_sess{s}.png")
            plt.close("all")
        return {}
    
class InferredPredsPlot:
    """Plot inferred rates for heldout neurons with optionally smoothed data."""
    def __init__(
        self,
        n_samples=4,
        n_batches=4,
        log_every_n_epochs=10,
        smooth=True,                # plot smoothed version of data (e.g. for spikes)
        plot_first_session=True,    # only plot results for the first session
        plot_random_units=False,    # if False, plots units with the highest activities
    ):
        self.name = "inferred_preds_plot"
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.log_every_n_epochs = log_every_n_epochs
        
        if smooth:
            self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        else:
            self.smoothing_func = lambda x: x
            
        self.plot_first_session = plot_first_session
        self.plot_random_units = plot_random_units

    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
        if len(pl_module.hparams.hn_indices) == 0: return
        
        # Get data and session
        batches, save_var = kwargs["batches"], kwargs["save_var"]
        if self.plot_first_session: sessions = [0]
        else: sessions = range(len(batches))
        
        for s in sessions: # holdout neuron indices are shared across sessions

            # Get data
            batch = batches[s]
            ic_enc_seq_len = pl_module.hparams.ic_enc_seq_len

            # Create subplots
            n_rows, n_cols = len(pl_module.area_names) * self.n_samples, self.n_batches
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                sharex=True,
                sharey="row",
                figsize=(3 * n_cols, 2 * n_rows),
            )
            common_label(fig, "time step", "rates")
            common_col_title(fig, [f"Batch {i}" for i in range(n_cols)], (n_rows, n_cols))
            axes = axes.reshape(n_rows, n_cols)

            # Iterate through areas and take n_sample neurons
            count = 0
            for area_name, area in pl_module.areas.items():
                if len(pl_module.hparams.hn_indices[area_name][s]) == 0:
                    continue
                
                encod_data = pl_module.holdout.hn_dict[area_name][:, ic_enc_seq_len:].cpu().detach().numpy()
                preds = pl_module.preds[area_name][s].detach().cpu()
                
                # Process outputs based on distribution
                if area.output_dist.name == "poisson":
                    infer_data = torch.exp(preds).numpy()
                else:
                    infer_data = preds.numpy()
                    
                units = range(min([self.n_samples, preds.shape[-1]]))
                for jn in units:

                    for ib in range(self.n_batches):
                        # Plot spikes at negative locations
                        smoothed_value = self.smoothing_func(encod_data[ib, :, jn])
                        y_lim_bottom = - max(smoothed_value) / 2
                        spike_idx = np.nonzero(encod_data[ib, :, jn])
                        spike_loc = np.ones(len(spike_idx)) * y_lim_bottom
                        axes[count][ib].plot(spike_idx, spike_loc, color="darkgray", marker=".")
                        
                        # Plot inferred and smooth data
                        axes[count][ib].plot(infer_data[ib, :, jn], "b")
                        axes[count][ib].plot(smoothed_value, "k--")

                    axes[count][0].set_ylabel(f"area {area_name}, neuron #{jn}")
                    count += 1

            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/inferred_preds_plot_epoch{trainer.current_epoch}_sess{s}.png")
            plt.close("all")
        return {}

class PSTHPlot:
    """Plot PSTH of neural activity according to different experimental conditions."""
    def __init__(self, n_samples=3, log_every_n_epochs=10, plot_first_session=True):
        self.name = "psth_plot"
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        self.plot_first_session = plot_first_session
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
        assert hasattr(pl_module, "conditions"), "MR-LFADS model needs to have 'conditions' attribute."
        
        # Get data and outputs
        batches, save_var, outputs = kwargs["batches"], kwargs["save_var"], kwargs["outputs"]
        if self.plot_first_session: sessions = [0]
        else: sessions = range(len(batches))
        
        for s in sessions:
            batch = batches[s]
            units = pl_module.maximum_activity_units(s, self.n_samples)
            categories, cond_indices = pl_module.conditions[s]
            ic_enc_seq_len = pl_module.hparams.ic_enc_seq_len

            # Create subplots
            n_rows, n_cols = len(pl_module.area_names) * self.n_samples, len(categories)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                sharex=True,
                sharey="row",
                figsize=(3 * n_cols, 2 * n_rows),
            )

            # For each condition (category):
            for ic, ax_col in enumerate(axes.T):
                count = 0
                included_batches = cond_indices[ic]

                # Iterate through areas and take n_sample neurons
                for area_name, area in pl_module.areas.items():
                    encod_data = batch.encod_data[area_name].detach().cpu().numpy()[:, ic_enc_seq_len:]
                    if area.output_dist.name == "poisson":
                        infer_data = torch.exp(outputs[area_name][s].detach().cpu()).numpy()
                    else:
                        infer_data = outputs[area_name][s].outputs.detach().cpu().numpy()

                    for jn in units[area_name]:
                        x_mean = self.smoothing_func(encod_data[included_batches, :, jn].mean(axis=0)) # shape = (T,)
                        r_mean = infer_data[included_batches, :, jn].mean(axis=0) # shape = (T,)
                        x_std = self.smoothing_func(encod_data[included_batches, :, jn].std(axis=0)) # shape = (T,)
                        r_std = infer_data[included_batches, :, jn].std(axis=0) # shape = (T,)

                        ax_col[count].plot(r_mean, "b")
                        ax_col[count].plot(x_mean, "k")
                        ax_col[count].plot(range(len(r_mean)), r_mean, "b")
                        ax_col[count].plot(range(len(x_mean)), x_mean, "k--")
                        ax_col[count].fill_between(range(len(r_mean)), r_mean - r_std, r_mean + r_std,
                                                   color="lightblue", alpha=0.5)
                        ax_col[count].fill_between(range(len(x_mean)), x_mean - x_std, x_mean + x_std,
                                                   color="gray", alpha=0.5)
                        ax_col[count].set_ylabel(f"{area_name}, neuron #{jn}")
                        ax_col[count].set_title(categories[ic].replace("_", ", "))
                        count += 1

            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/psth_plot_epoch{trainer.current_epoch}_sess{s}.png")
            plt.close("all")
        return {}
    
# ========== Monitor/Summary Plots ========== #

def batch_smoothing_func(x):
    smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
    return np.apply_along_axis(smoothing_func, axis=1, arr=x)

def batch_corrcoef(x, y):
    N = x.shape[2]
    x_reshape = x.reshape(-1, N) # shape = (B*T, N)
    y_reshape = y.reshape(-1, N)
    return [np.corrcoef(x_reshape[:, i], y_reshape[:, i])[0][1] for i in range(N)]

class ProctorSummaryPlot:
    """Plot loss metrics and coefficients."""
    def __init__(self, log_every_n_epochs=10):
        self.name = "proctor_summary_plot"
        self.log_every_n_epochs = log_every_n_epochs
        self.count = 0
        self.corrs = {}
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if self.count < 2:
            self.count += 1
            for area_name in pl_module.areas: self.corrs[area_name] = []
            return
        
        # Uses just the first session
        s = 0
        
        # Access hyperparameters
        hps = pl_module.hparams
        epochs = np.arange(0, trainer.max_epochs)
        log_metrics = kwargs["log_metrics"]
        batches, save_var, outputs = kwargs["batches"], kwargs["save_var"], kwargs["outputs"]
        seq_len = hps.seq_len - hps.ic_enc_seq_len
    
        # Create subplots
        n_rows, n_cols = 4, 2
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=False,
            sharey=False,
            figsize=(3 * n_cols, 2 * n_rows),
        )
        common_label(fig, "epochs", "")
        
        # Plot lowest possible learning rate
        axes[0][0].plot(log_metrics["lr-AdamW"][1:], "k")
        axes[0][0].set_xlabel("steps")
        
        # Plot KL divergence ramp history
        axes[0][1].plot(log_metrics["valid/kl/ramp/u"][1:], "k", label="u")
        axes[0][1].plot(log_metrics["valid/kl/ramp/m"][1:], "b--", label="m")
        axes[0][1].plot(log_metrics["valid/kl/ramp/g"][1:], "g--", label="g")
        axes[0][1].legend()
        
        num_areas = len(pl_module.areas)
        for ia, (area_name, area) in enumerate(pl_module.areas.items()):
            
            # Compute correlation
            true_data = batches[s].encod_data[area_name][:, hps.ic_enc_seq_len:].cpu().detach().numpy()
            if area.output_dist.name == "poisson":
                pred_rates = torch.exp(outputs[area_name][s].cpu().detach()).numpy()
            elif area.output_dist.name == "gaussian":
                nn = area.hparams.num_neurons[s]
                pred_rates = outputs[area_name][s].cpu().detach().numpy()[..., :nn]
            else:
                pred_rates = outputs[area_name][s].cpu().detach().numpy()
            avg_rates = true_data.mean(axis=(0,1))
            smoothed_rates = batch_smoothing_func(true_data)
            corrs = batch_corrcoef(smoothed_rates, pred_rates)
            self.corrs[area_name].append(np.mean(corrs))
            
            comb = np.array(log_metrics[f"valid/l1"])/num_areas + np.array(log_metrics[f"valid/{area_name}/l2"])
            
            axes[1][0].plot(log_metrics[f"valid/{area_name}/recon"][1:], label=area_name)
            axes[1][1].plot(log_metrics[f"valid/{area_name}/hn"][1:], label=area_name)
            axes[2][0].plot(log_metrics[f"valid/{area_name}/kl/co"][1:], label=area_name)
            axes[2][1].plot(log_metrics[f"valid/{area_name}/kl/com"][1:], label=area_name)
            axes[3][0].plot(log_metrics[f"valid/{area_name}/kl/gv"][1:], label=area_name)
            axes[3][1].plot(comb[1:], label=area_name)
            
        axes[0][0].set_title("Learning Rate History")
        axes[0][1].set_title("KL Coefficient History")
        axes[1][0].set_title("Reconstruction Loss")
        axes[1][1].set_title("Holdout Recon Loss") 
        axes[2][0].set_title("KL Divergence Loss (u)")
        axes[2][1].set_title("KL Divergence Loss (m)")
        axes[3][0].set_title("KL Divergence Loss (g)")
        axes[3][1].set_title("L1 + L2 Loss")
        
        axes[1][0].legend()
        axes[1][1].legend()
        axes[2][0].legend()
        axes[2][1].legend()
        axes[3][0].legend()
        axes[3][1].legend()
        axes[0][0].set_ylabel("learning rate")
        axes[0][1].set_ylabel("KL divergence")
        axes[1][0].set_ylabel("loss") 
        axes[2][0].set_ylabel("loss") 
        axes[3][0].set_ylabel("loss") 
        axes[3][1].set_ylabel("loss") 
        
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/proctor_summary_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
    
# ========== Communication Plots ========== #

class CommunicationPSTHPlot:
    """Plot Inferred Input and Communication PSTH plots for all areas."""
    def __init__(self, log_every_n_epochs=10, var_name="batch"):
        self.name = "communication_psth_plot"
        self.log_every_n_epochs = log_every_n_epochs
        self.count = 0
        self.var_name = var_name
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if self.count < 2:
            self.count += 1
            return
        assert hasattr(pl_module, "conditions"), "MR-LFADS model needs to have 'conditions' attribute."
        
        # Use just the first session
        s = 0
        
        # Get data and outputs
        batches, save_var = kwargs["batches"], kwargs["save_var"]
        log_metrics = kwargs["log_metrics"]
        cmap = sns.color_palette("viridis", as_cmap=True)
        batch = batches[s]
        categories, cond_indices = pl_module.conditions[s]
        
        # Get batch size
        fix = pl_module.current_info[0][self.var_name]
        batch_size = len(fix)

        # Create subplots
        n_rows, n_cols = len(pl_module.area_names) * 4, len(categories)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=True,
            sharey=False,
            figsize=(3 * n_cols, 2 * n_rows),
        )
        common_col_title(fig, categories, (n_rows, n_cols))
        axes = axes.reshape((n_rows, n_cols))

        # For each condition (category):
        for ic, ax_col in enumerate(axes.T):
            count = 0
            included_batches = cond_indices[ic]
            included_batches = included_batches[included_batches < batch_size]

            # Iterate through areas and take n_sample neurons
            for ia, (area_name, area) in enumerate(pl_module.areas.items()):
                hps = area.hparams
                num_other_areas_name = list(pl_module.areas.keys())
                num_other_areas_name.pop(ia)
                inputs = save_var[area_name].inputs.detach().cpu()
                ci_size, com_dim, co_dim = hps.ci_size, hps.com_dim, hps.co_dim
                _, com, co = torch.split(inputs, [ci_size, com_dim * hps.num_other_areas, co_dim], dim=2)

                # Get colors
                colors = plt.cm.rainbow(np.linspace(0, 1, hps.num_other_areas))

                if area.hparams.use_con:
                    # Plot co
                    for ico in range(co_dim):
                        ax_col[count].plot(co[included_batches, :, ico].mean(axis=0))
                    ax_col[count].set_ylabel(f"{area_name}, u")
                    count += 1

                    # Plot kl (co)
                    co_mean, co_std = torch.split(save_var[area_name].co_params, [hps.co_dim, hps.co_dim], dim=2)
                    co_kl = area.co_prior(co_mean[included_batches], co_std[included_batches], 'seq', 1)
                    for jco in range(co_dim):
                        ax_col[count].plot(co_kl[jco].cpu().detach().numpy())
                    ax_col[count].set_ylabel(f"{area_name}, kl (u)")
                    count += 1
                else:
                    count += 2

                # Plot com
                count_com = 0
                for icom in range(hps.num_other_areas):
                    for ii in range(com_dim):
                        perturbation = np.random.uniform(-0.25, 0.25, size=3)
                        perturbed_color = np.clip(colors[icom][:3] + perturbation, 0.0, 1.0)
                        sub_color = (*perturbed_color, (ii + 1) / (com_dim + 1)) # colors[icom] is the group color
                        if (ii == 0) and (ic == 0):
                            ax_col[count].plot(com[included_batches, :, count_com].mean(axis=0), color=sub_color, label=f"{num_other_areas_name[icom]}")
                        else:
                            ax_col[count].plot(com[included_batches, :, count_com].mean(axis=0), color=sub_color)
                        count_com += 1
                ax_col[count].set_ylabel(f"{area_name}, m")
                count += 1

                # Plot kl (com)
                com_mean, com_std = torch.split(save_var[area_name].com_params, [hps.com_dim * hps.num_other_areas, hps.com_dim * hps.num_other_areas], dim=2)
                com_kl = area.com_prior(com_mean[included_batches], com_std[included_batches], 'seq', 1)
                count_kl = 0
                for jcom in range(hps.num_other_areas):
                    for jj in range(com_dim):
                        perturbation = np.random.uniform(-0.25, 0.25, size=3)
                        perturbed_color = np.clip(colors[jcom][:3] + perturbation, 0.0, 1.0)
                        sub_color = (*perturbed_color, (jj + 1) / (com_dim + 1)) # colors[jcom] is the group color
                        if (jj == 0) and (ic == 0):
                            ax_col[count].plot(com_kl[count_kl].cpu().detach().numpy(), color=sub_color, label=f"{num_other_areas_name[icom]}")
                        else:
                            ax_col[count].plot(com_kl[count_kl].cpu().detach().numpy(), color=sub_color)
                        count_kl += 1
                ax_col[count].set_ylabel(f"{area_name}, kl (m)")
                count += 1

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/communication_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
    
def plot_anatomy(model, ax=None, save=None, last=None, co_scale=1):   
    kl_weight_dict = {}
    num_areas = len(model.areas)
    if num_areas < 2: return
    
    model.to("cuda")
    for area_name, area in model.areas.items():
        if area.hparams.use_con:
            co_mean, co_std = torch.split(model.save_var[area_name].co_params, area.hparams.co_dim, dim=2)
            co_kl = np.log(area.co_prior(co_mean, co_std).item()) * co_scale # scale is already multiplied
        else:
            co_kl = 0
            
        com_mean, com_std = torch.split(model.save_var[area_name].com_params, area.hparams.com_dim * (num_areas-1), dim=2)
        com_kl = np.log(np.array(area.com_prior(com_mean, com_std, 'mean', area.hparams.com_dim)))

        other_areas = model.area_names.copy()
        other_areas.remove(area_name)
        for io, other_area in enumerate(other_areas):
            kl_weight_dict[(other_area[:last], area_name[:last])] = com_kl[io]
            
        # Inferred inputs
        kl_weight_dict[("Other", area_name[:last])] = co_kl
        
        # Global area
        if model.hparams.gv_dim > 0:
            gv_mean, gv_std = torch.split(model.save_var[area_name].gv_params, model.hparams.gv_dim, dim=2)
            gv_mean = area.decoder.gv_scalar.reshape(1, 1, -1).tile(*gv_mean.shape[:2], 1) * gv_mean
            gv_kl = np.log(model.global_area.gv_prior(gv_mean, gv_std).item())
            kl_weight_dict[("Global", area_name[:last])] = gv_kl
        
    # Create Digraph
    color_choice = ["black", "red", "green"]
    G = nx.DiGraph()
    for edge, weight in kl_weight_dict.items():
        G.add_edge(edge[0], edge[1], weight=weight)
    pos = nx.circular_layout(G)
    edges = G.edges()
    weights = np.array([kl_weight_dict[edge] for edge in edges])
    colors = np.array([color_choice[
        int(("Other" in edge) or ("Global" in edge)) * (int("Global" in edge) + 1)
    ] for edge in edges])
    
    # Normalize and scale weights
    weights -= np.min(weights)
    weights /= np.max(weights)
    weights += 0.1
    weights *= 5

    if not ax: fig, ax = plt.subplots(1, 1)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_weight='bold', connectionstyle='arc3, rad = 0.1', width=weights, edge_color=colors, ax=ax)
    plt.tight_layout()
    
    if save: plt.savefig(f"/root/capsule/results/{save}.png")
    return kl_weight_dict
    
class AnatomyPlot:
    """Plot inferred effectome based on communication message norms."""
    def __init__(self, log_every_n_epochs=10):
        self.name = "anatomy_plot"
        self.log_every_n_epochs = log_every_n_epochs

    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
        
        plot_anatomy(pl_module)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/anatomy_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
    
# ========== Other Plots ========== #
        
class ICPlot:
    def __init__(self, log_every_n_epochs=10):
        self.name = "ic_plot"
        self.log_every_n_epochs = log_every_n_epochs

    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        # Get data
        batches, save_var = kwargs["batches"], kwargs["save_var"]
        batch = batches[0]
        hps = pl_module.hparams
        
        # Setup
        num_cols = max([len(pl_module.areas), 2])
        fig, axs = plt.subplots(1, num_cols, figsize=(2*num_cols, 2))
        
        for ia, area_name in enumerate(pl_module.areas):
            preds = pl_module.outputs[area_name][0][:, 0, :].cpu().detach().numpy()
            trues = batch.encod_data[area_name].detach().cpu().numpy()[:, hps.ic_enc_seq_len]
            axs[ia].scatter(trues.reshape(-1), preds.reshape(-1), c="k")
            
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/ic_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
        
class ProctorPreviewPlot:
    def __init__(self,):
        self.name = "proctor_preview_plot"
    
    def run(self, trainer, pl_module, **kwargs):
        # Access hyperparameters
        hps = pl_module.hparams
        epochs = np.arange(0, trainer.max_epochs)
        batches = kwargs["batches"]

        # Create subplots
        n_rows, n_cols = 3, 1
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex=False,
            sharey="row",
            figsize=(3 * n_cols, 2 * n_rows),
        )

        # Plot lowest possible learning rate
        geom = lambda epoch: hps.lr_init * np.power(hps.lr_decay, epoch // hps.lr_patience)
        axes[0].plot(epochs, geom(epochs), "k")
        axes[0].set_title(f"Lowest lr: {round(geom(trainer.max_epochs) ,6)}")
        axes[0].set_ylabel("learning rate")
        axes[0].set_xlabel("epoch")

        # Plot KL divergence history
        kl_ramp_u = pl_module._compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_co, hps.kl_increase_epoch_co)
        kl_ramp_m = pl_module._compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_com, hps.kl_increase_epoch_com)
        kl_ramp_g = pl_module._compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_gv, hps.kl_increase_epoch_gv, hps.kl_init_gv_scale)
        axes[1].plot(kl_ramp_u, "k", label="u")
        axes[1].plot(kl_ramp_m, "b--", label="m")
        axes[1].plot(kl_ramp_g, "g--", label="g")
        axes[1].set_ylabel("KL divergence")
        axes[1].set_xlabel("epoch")
        axes[1].set_title("KL Divergence History")
        axes[1].legend()

        # Plot external input
        for s in range(len(batches)):
            for area_name in batches[s].ext_input.keys():
                arr = batches[s].ext_input[area_name]
                if (len(arr.shape) == 3) or (arr.shape[-1] != 0): 
                    photostim_batches = np.where(arr.mean(axis=(1,2)) != 0)[0]
                    for b in photostim_batches[:5]: axes[2].plot(arr[b].squeeze())

        axes[2].set_ylabel("amplitude")
        axes[2].set_xlabel("time step")
        axes[2].set_title("External Input")

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/proctor_preview.png")
    
# ========== Pre-processing functions ========== #

class get_maximum_activity_units:
    def __init__(self,):
        self.name = "get_maximum_activity_units"
    
    def run(self, trainer, pl_module, **kwargs):
        batches = kwargs["batches"]

        session_units = []
        for s in range(len(batches)):
            units = {}
            batch = batches[s]
            for area_name in pl_module.area_names:
                arr = batch.encod_data[area_name].detach().cpu().numpy() # shape = (B, T, N)
                arr = arr.reshape(-1, arr.shape[-1]) # shape = (B*T, N)
                arr = np.abs(arr) # for arrays that have negative values (not spike trains)
                indices = np.flip(np.argsort(arr.mean(0))) # according to mean across batch, time
                units[area_name] = indices
            session_units.append(units)
        pl_module.maximum_activity_units = lambda s, n_samples: {k: v[:n_samples] for k, v in session_units[s].items()}

class get_default_conditions:
    def __init__(self, var_name="batch"):
        self.name = "default_condition"
        self.var_name = var_name
    
    def run(self, trainer, pl_module, **kwargs):
        info_dict = kwargs["info"][0]
        print(info_dict.keys())
        batch_size = info_dict[self.var_name].shape[0]
        indices = [np.arange(batch_size).astype(int)]
        categories = [0]
        
        pl_module.conditions = {0: (categories, indices)}
        
class get_conditions:
    def __init__(self, var_name="batch"):
        self.name = "get_conditions"
        self.var_name = var_name

    def run(self, trainer, pl_module, **kwargs):
        batches, info_dict = kwargs["batches"], kwargs["info"]

        conditions = []
        for s in range(len(info_dict)):
            info_strings = info_dict[s][self.var_name]
            categories, inverse_indices = np.unique(info_strings, return_inverse=True)
            unique_indices = [np.where(inverse_indices == i)[0] for i in range(len(categories))]
            conditions.append( (categories, unique_indices) )
        pl_module.conditions = conditions 