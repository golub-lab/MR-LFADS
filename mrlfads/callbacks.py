"""
Callback groups for training and validation.

Main classes
------------
OnEpochEndCalls:
    Run callback objects at epoch end for training and/or validation.
"""

import io
import os
import math
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import mrlfads.utils.visualization_utils as vis
import mrlfads.utils.dir_utils as nav
from mrlfads.evals.visualization import plot_anatomy
from mrlfads.utils.common_utils import batch_smoothing_func

plt.switch_backend("Agg")
SAVE_DIR = "./graphs"

# ===== Main class ===== #

class OnEpochEndCalls(pl.Callback):
    def __init__(self,
                 callbacks: list,
                 priority: int = 1):
        self.priority = priority
        self.callbacks = callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def run(self, trainer, pl_module, step_type):
        kwargs = {"step_type": step_type}
        for i, callback in enumerate(self.callbacks):
            # Use log if present as kwargs
            if callback.name == "log":
                kwargs["metrics"] = callback.run(trainer, pl_module)
            
            # run callback according to step_type
            if step_type in callback.run_steps:
                new_kwargs = callback.run(trainer, pl_module, **kwargs)
                
    def on_train_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "train")
            
    def on_validation_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "valid")
    
# ===== Logs, metrics ===== #

class Log:
    def __init__(self,
                 tags: list = [],
                 run_steps: list = ["train", "valid"]
                ):
        self.name = "log"
        self.run_steps = run_steps
        self.metrics = defaultdict(list)
        self.tags = tags
    
    def run(self, trainer, pl_module, **kwargs):
        new_metrics = trainer.logged_metrics
        self._update_dict(self.metrics, new_metrics)
        
        # Find tensorboard logger
        tb_logger = next(
            (l for l in trainer.loggers if isinstance(l, TensorBoardLogger)),
            None,
        )
        assert tb_logger is not None, "TensorBoardLogger not found in trainer.loggers."
        log_dir = tb_logger.log_dir
        
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        for tag in self.tags:
            if tag in event_acc.Tags()["scalars"]:
                scalar_events = event_acc.Scalars(tag)
                values = [event.value for event in scalar_events]
                self.metrics[tag] = values
            else:
                self.metrics[tag] = []
        return self.metrics
        
    def _update_dict(self, old_dict, new_dict):
        for key, value in new_dict.items():
            old_dict[key].append(value.item())
            
class ProctorSummaryPlot:
    def __init__(self, log_every_n_epochs=10, run_steps=["valid"]):
        self.name = "proctor_summary_plot"
        self.run_steps = run_steps
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
        log_metrics = kwargs["metrics"]
        batches, save_var, outputs = pl_module.current_batch, pl_module.save_var, pl_module.outputs
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
        vis.common_label(fig, "epochs", "")
        
        # Plot lowest possible learning rate
        axes[0][0].plot(log_metrics["lr-AdamW"][1:], "k")
        axes[0][0].set_title("Learning Rate History")
        axes[0][0].set_ylabel("learning rate")
        axes[0][0].set_xlabel("steps")
        
        # Plot KL divergence ramp history
        axes[0][1].plot(log_metrics["valid/kl/ramp/u"][1:], "k", label="u")
        axes[0][1].plot(log_metrics["valid/kl/ramp/m"][1:], "b--", label="m")
        axes[0][1].plot(log_metrics["valid/kl/ramp/g"][1:], "g--", label="g")
        axes[0][1].set_ylabel("KL divergence")
        axes[0][1].set_title("KL Coefficient History")
        axes[0][1].legend()
        
        num_areas = len(pl_module.areas)
        for ia, (area_name, area) in enumerate(pl_module.areas.items()):
            
            axes[1][0].plot(log_metrics[f"valid/{area_name}/recon"][1:], label=area_name)
            axes[1][1].plot(log_metrics[f"valid/{area_name}/hn"][1:], label=area_name)
            axes[2][0].plot(log_metrics[f"valid/{area_name}/kl/co"][1:], label=area_name)
            axes[2][1].plot(log_metrics[f"valid/{area_name}/kl/com"][1:], label=area_name)
            axes[3][0].plot(log_metrics[f"valid/{area_name}/kl/gv"][1:], label=area_name)
            axes[3][1].plot(log_metrics[f"valid/{area_name}/l2"][1:], label=area_name)
            
        axes[1][0].set_title("Reconstruction Loss")
        axes[1][0].legend(fontsize=6)
        axes[1][1].set_title("Holdout Recon Loss")
        axes[1][1].legend(fontsize=6)
        axes[2][0].set_title("KL Divergence Loss (u)")
        axes[2][0].legend(fontsize=6)
        axes[2][1].set_title("KL Divergence Loss (m)")
        axes[2][1].legend(fontsize=6)
        axes[3][0].set_title("KL Divergence Loss (g)")
        axes[3][0].legend(fontsize=6)
        axes[3][1].set_title("L2 Loss") 
        axes[3][1].legend(fontsize=6)
        axes[1][0].set_ylabel("loss") 
        axes[2][0].set_ylabel("loss") 
        axes[3][0].set_ylabel("loss") 
        
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/proctor_summary_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
    
class BasicProctorSummaryPlot:
    def __init__(self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
        multiple_acc: bool = False,
    ):
        self.name = "basic_proctor_summary_plot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.multiple_acc = multiple_acc
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return

        metrics = kwargs["metrics"]
        num_rows, num_cols = 2, 2

        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*3),
            sharex = True,
            sharey = False,
        )
        vis.common_row_ylabel(fig, ["loss", "accuracy"], (num_rows, num_cols))
        
        i_train, i_valid = 0, 0
        colors = sns.color_palette("Set2", len(metrics))
        for key, value in metrics.items():
            step_type, name = key.split("/")
            if ('loss' not in name) and ('acc' not in name): continue # skip if not loss or acc
            
            is_loss = "loss" in name
            is_valid = step_type == "valid"
            idx = i_valid if is_valid else i_train
            axs[1-int(is_loss)][int(is_valid)].plot(value, label=name[:10], color=colors[idx])
            axs[1-int(is_loss)][int(is_valid)].set_title(step_type)
            if is_loss or self.multiple_acc: axs[1-int(is_loss)][int(is_valid)].legend(fontsize='small')

            if is_valid: i_valid += 1
            else: i_train += 1

        plt.tight_layout()
        vis.savefig(f"BasicProctorSummary_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)
    
class proctor_preview_plot:
    def __init__(self, run_steps=["valid"]):
        self.name = "proctor_preview_plot"
        self.run_steps = run_steps
        self.ran = False
    
    def run(self, trainer, pl_module, **kwargs):
        if self.ran:
            return
        else:
            self.ran = True
            
        # Access hyperparameters
        hps = pl_module.hparams
        epochs = np.arange(0, trainer.max_epochs)
        batches = pl_module.current_batch

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
        kl_ramp_u = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_co, hps.kl_increase_epoch_co)
        kl_ramp_m = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_com, hps.kl_increase_epoch_com)
        kl_ramp_g = pl_module.compute_ramp_inner(torch.from_numpy(epochs), hps.kl_start_epoch_gv, hps.kl_increase_epoch_gv)
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

                if batches[s].ext_input[area_name].shape[2] > 0:
                    arr = batches[s].ext_input[area_name].cpu().detach().numpy()
                    photostim_batches = np.where(arr.mean(axis=(1,2)) != 0)[0]
                    for b in photostim_batches[:5]: axes[2].plot(arr[b].squeeze())
        axes[2].set_ylabel("amplitude")
        axes[2].set_xlabel("time step")
        axes[2].set_title("External Input")

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/proctor_preview.png")

class TimerPlot:
    def __init__(self, log_every_n_epochs=20, run_steps=["train", "valid"]):
        self.name = "timer_plot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.cur_time = time.perf_counter()
        self.elapsed = []

    def run(self, trainer, pl_module, **kwargs):

        prev_time = self.cur_time
        now = time.perf_counter()
        elapsed = now - prev_time
        self.cur_time = now
        self.elapsed.append(elapsed)

        message = f"EPOCH {trainer.current_epoch}: {elapsed}"
        nav.write_logfile("timer.txt", message)
        
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            fig, ax = plt.subplots(1, 1, figsize=(4,3))
            plt.plot(self.elapsed, 'k.')
            plt.xlabel('Records')
            plt.ylabel('Elapsed Time')
            vis.set_invisible(ax)
            plt.savefig(f"{SAVE_DIR}/timer_plot_epoch{trainer.current_epoch}.png")
            plt.close("all")
        return {}
    
# ===== Visualizations ===== #
    
class InferredRatesPlot:
    """
    Plots inferred rates with smoothed spiking data.
    """
    def __init__(self, n_samples=3, n_batches=4, log_every_n_epochs=10, plot_first_session=True, smooth=True, plot_random_units=False, run_steps=["valid"]):
        self.name = "inferred_rates_plot"
        self.run_steps = run_steps
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
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        # Get data and session
        batches, save_var, outputs = pl_module.current_batch, pl_module.save_var, pl_module.outputs
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
            vis.common_label(fig, "time step", "rates")
            vis.common_col_title(fig, [f"Batch {i}" for i in range(n_cols)], (n_rows, n_cols))

            # Iterate through areas and take n_sample neurons
            count = 0
            for area_name, area in pl_module.areas.items():
                encod_data = batch.encod_data[area_name].detach().cpu().numpy()[:, ic_enc_seq_len:]
                if area.output_dist.name == "poisson":
                    infer_data = torch.exp(outputs[area_name][s].detach().cpu()).numpy()
                else:
                    infer_data = outputs[area_name][s].detach().cpu().numpy()
                    
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
    """
    Plots inferred rates with smoothed spiking data.
    """
    def __init__(self, n_samples=4, n_batches=4, log_every_n_epochs=10, plot_first_session=True, smooth=True, plot_random_units=False, run_steps=["valid"]):
        self.name = "inferred_preds_plot"
        self.run_steps = run_steps
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
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if len(pl_module.hparams.hn_indices) == 0:
            return
        
        # Get data and session
        batches, save_var = pl_module.current_batch, pl_module.save_var
        if self.plot_first_session: sessions = [0]
        else: sessions = range(len(batches))
        
        for s in sessions:

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
            vis.common_label(fig, "time step", "rates")
            vis.common_col_title(fig, [f"Batch {i}" for i in range(n_cols)], (n_rows, n_cols))
            axes = axes.reshape(n_rows, n_cols)

            # Iterate through areas and take n_sample neurons
            count = 0
            for area_name, area in pl_module.areas.items():
                encod_data = pl_module.holdout.hn_dict[s][area_name][:, ic_enc_seq_len:].cpu().detach().numpy()
                preds = pl_module.preds[area_name][s].detach().cpu()
                
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
    """
    Plot PSTH for all areas.
    """
    def __init__(self, n_samples=3, log_every_n_epochs=10, plot_first_session=True, run_steps=["valid"]):
        self.name = "psth_plot"
        self.run_steps = run_steps
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.smoothing_func = lambda x: gaussian_filter1d(x.astype(float), sigma=10)
        self.plot_first_session = plot_first_session
        
    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        # Get data and outputs
        batches, save_var, outputs = pl_module.current_batch, pl_module.save_var, pl_module.outputs
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

                    if area.output_dist.name == "zipoisson":
                        non_zero_prob = 1 - area.output_dist.zero_prob.detach().cpu().numpy()
                    else:
                        non_zero_prob = np.ones(infer_data.shape[-1])

                    for jn in units[area_name]:
                        x_mean = self.smoothing_func(encod_data[included_batches, :, jn].mean(axis=0)) # shape = (T,)
                        r_mean = infer_data[included_batches, :, jn].mean(axis=0) # shape = (T,)
                        x_std = self.smoothing_func(encod_data[included_batches, :, jn].std(axis=0)) # shape = (T,)
                        r_std = infer_data[included_batches, :, jn].std(axis=0) # shape = (T,)

                        r_mean *= non_zero_prob[jn]
                        r_std *= abs(non_zero_prob[jn])
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

class CommunicationPSTHPlot:
    """
    Plot Inferred Input and Communication PSTH plots for all areas.
    """
    def __init__(self, log_every_n_epochs=10, var_name="fix", run_steps=["valid"]):
        self.name = "communication_psth_plot"
        self.run_steps = run_steps
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
        if pl_module.hparams.num_other_areas == 0: return
        
        # Use just the first session
        s = 0
        
        # Get data and outputs
        batches, save_var = pl_module.current_batch, pl_module.save_var
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
        vis.common_col_title(fig, categories, (n_rows, n_cols))
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
                ci_enc_dim, com_dim, co_dim = hps.ci_enc_dim, hps.com_dim, hps.co_dim
                _, com, co = torch.split(inputs, [ci_enc_dim, com_dim * hps.num_other_areas, co_dim], dim=2)

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
                    co_kl = area.co_prior.kl_divergence_by_component(co_mean[included_batches], co_std[included_batches], 1, tpe="seq")
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
                com_kl = area.com_prior.kl_divergence_by_component(com_mean[included_batches], com_std[included_batches], 1, tpe="seq")
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
    
class DecoderPlot:
    def __init__(self,
        log_every_n_epochs: int = 1,
        run_steps: list = ["valid"],
        n_batches: int = 4,
    ):
        self.name = "decoder_plot"
        self.run_steps = run_steps
        self.n_batches = n_batches
        self.log_every_n_epochs = log_every_n_epochs
    
    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return

        hps = pl_module.hparams
        
        num_rows, num_cols = len(hps.predictions), self.n_batches
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(num_cols*3, num_rows*3),
            sharex = True,
            sharey = False,
        )
        axs = axs.reshape(num_rows, num_cols)
        
        for i, (str_tuple, output) in enumerate(pl_module.outputs.items()):
            for b in range(self.n_batches):
                axs[i, b].plot(pl_module.targets.cpu().detach().numpy()[b, :, 0], 'k.')
                axs[i, b].plot(torch.sigmoid(output).cpu().detach().numpy()[b, :, 0], 'b.')
                
        vis.common_col_title(fig, [f"Batch {i}" for i in range(num_cols)], axs.shape)
        vis.common_row_ylabel(fig, [f"Pred {i}" for i in range(num_rows)], axs.shape)
        vis.savefig(f"Decode_epoch={trainer.current_epoch}.png", folders=[SAVE_DIR], close=True)
    
# ===== Inference ===== #
    
class AnatomyPlot:
    """
    Plots inferred rates with smoothed spiking data.
    """
    def __init__(self, log_every_n_epochs=10, run_steps=["valid"]):
        self.name = "anatomy_plot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs

    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if pl_module.hparams.num_other_areas == 0: return
        
        plot_anatomy(pl_module)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/anatomy_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return {}
        
class ICPlot:
    def __init__(self, log_every_n_epochs=10, run_steps=["valid"]):
        self.name = "ic_plot"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs

    def run(self, trainer, pl_module, **kwargs):
        # Check for conditions to not run
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        
        # Get data
        batches, save_var = pl_module.current_batch, pl_module.save_var
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
        
# ===== Helpers ===== #
        
class get_maximum_activity_units:
    def __init__(self, run_steps=["valid"]):
        self.name = "get_maximum_activity_units"
        self.run_steps = run_steps
        self.ran = False
    
    def run(self, trainer, pl_module, **kwargs):
        if self.ran:
            return
        else:
            self.ran = True
            
        batches = pl_module.current_batch
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
    
class get_conditions:
    def __init__(self, run_steps=["valid"]):
        self.name = "get_conditions"
        self.run_steps = run_steps
        self.ran = False

    def run(self, trainer, pl_module, **kwargs):
        if self.ran:
            return
        else:
            self.ran = True
            
        batches, info_dict = pl_module.current_batch, pl_module.current_info

        conditions = []
        for s in range(len(info_dict)):
            info_strings = info_dict[s]["instruction_outcome"]
            categories, inverse_indices = np.unique(info_strings, return_inverse=True)
            unique_indices = [np.where(inverse_indices == i)[0] for i in range(len(categories))]
            conditions.append( (categories, unique_indices) )
        pl_module.conditions = conditions 
        
class get_default_conditions:
    def __init__(self, var_name="batch", run_steps=["valid"]):
        self.name = "default_condition"
        self.run_steps = run_steps
        self.var_name = var_name
        self.ran = False
    
    def run(self, trainer, pl_module, **kwargs):
        if self.ran:
            return
        else:
            self.ran = True
            
        info_dict = pl_module.current_info[0]
        print(info_dict.keys())
        batch_size = info_dict[self.var_name].shape[0]
        indices = [np.arange(batch_size).astype(int)]
        categories = [0]
        
        pl_module.conditions = {0: (categories, indices)}

class ComputeInitWeights:
    def __init__(self,
                 normalize: bool = True,
                 filter_cond: bool = True,
                 run_steps: list = ["valid"],
                ):
        self.name = "compute_init_weights"
        self.normalize = normalize
        self.filter_cond = filter_cond
        self.run_steps = run_steps
        self.ran = False
    
    def run(self, trainer, pl_module, **kwargs):
        if self.ran:
            return
        else:
            self.ran = True
            
        dm = trainer.datamodule
        all_session_datasets = (
            dm.train_session_datasets + dm.val_session_datasets
        )
        idx_sessions = list(range(len(dm.train_session_datasets))) + \
        list(range(len(dm.val_session_datasets)))
        
        area_arr_dict = {an: [] for an in pl_module.area_names}
        cond_arr = []
        sess_arr = []
        
        for i, session_ds in enumerate(all_session_datasets):
            for batch_idx in session_ds.indices:
                
                # Get spike count data
                dic = session_ds.dataset.data_list[batch_idx]
                # for area_name, arr in dic.items():
                for area_name in pl_module.area_names:
                    arr = dic[area_name]
                    area_arr_dict[area_name].append(arr)
                    
                # Get condition
                info = session_ds.dataset.info_list[batch_idx]
                cond = info["instruction_outcome"]
                cond_arr.append(cond)
                
                # Get session
                sess_arr.append(idx_sessions[i])
               
        # Find conditions to use 
        conds = list(set(cond_arr))
        if self.filter_cond: # drop if < 5%
            uniq, counts = np.unique(cond_arr, return_counts=True)
            conds = uniq[counts > len(cond_arr) * 0.05]
        cond_arr = np.array(cond_arr)
        
        # Get all sessions
        sessions = np.sort(np.unique(sess_arr))
        sess_arr = np.array(sess_arr)
        
        # Main
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        palette = sns.color_palette('husl', len(pl_module.areas))
        
        weights_dict = {}
        for ia, area_name in enumerate(area_arr_dict):
            
            # Compute PSTH
            psths = {} # session --> psths
            exist = [True] * len(conds)
            for sess in sessions:
                idx_sess = np.where(sess_arr == sess)[0]
                
                temp = []
                for ic, cond in enumerate(conds):
                    idx_cond = np.where(cond_arr == cond)[0]
                    indices = np.intersect1d(idx_sess, idx_cond)
                    if len(indices) > 0:
                        data = np.stack(
                            [area_arr_dict[area_name][idx] for idx in indices],
                            axis=0
                        ) # (batch, time, neu)
                        temp.append( data.mean(axis=0) ) # shape = (time, neu)
                    else:
                        temp.append(None)
                        exist[ic] = False
                
                psths[sess] = temp
                
            # Concatenate
            for sess in sessions:
                temp = [psths[sess][ic] for ic in range(len(conds)) if exist[ic]]
                psths[sess] = np.concatenate(temp, axis=0) # shape = (#cond x time, neu)
            
            # Get PCA
            data_dim = pl_module.areas[area_name].hparams.data_dim
            pca = PCA(data_dim)
            psths_all_session = np.concatenate([psths[s] for s in sessions], axis=1) # (#cond x time, total neu)
            transformed = pca.fit_transform(psths_all_session) # shape = (#cond x time, data_dim)
            
            # Plot PCA
            evr = np.cumsum(pca.explained_variance_ratio_)
            ax.plot(evr, c=palette[ia], marker='.', label=area_name)
            
            # Regress for each session
            reg = LinearRegression(fit_intercept=False)
            weights_dict[area_name] = {}
            offset = 0
            for sess in sessions:
                
                # Centering
                X = psths[sess]
                D_s = X.shape[1]
                mean_slice = pca.mean_[offset:offset + D_s]
                Xc = X - mean_slice
                
                reg.fit(Xc, transformed)
                offset += D_s
                
                w = weights_dict[area_name][sess] = reg.coef_ # shape = (data_dim, neu)
                
                # Initialize readin, readout weights
                readin = pl_module.areas[area_name].readin
                readout = pl_module.areas[area_name].readout
                w = torch.from_numpy(w).to(device=readin[sess].weight.device,
                           dtype=torch.float32)

                # Normalize
                sigma_pcr = w.std().clamp_min(1e-8) # guard small std
                sigma_target = 1.0 / math.sqrt(w.shape[1])
                if self.normalize:
                    w = w * (sigma_target / sigma_pcr)
                w = w.to(dtype=readin[sess].weight.dtype)

                with torch.no_grad():
                    readin[sess].weight.copy_(w)
                    readout[sess].weight.copy_(w.t())
                
        vis.set_invisible(ax)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/init_pca_plot_epoch{trainer.current_epoch}.png")
        plt.close("all")
        return weights_dict
    
