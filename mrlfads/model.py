import os
import torch
import numpy as np
import pytorch_lightning as pl

from torch import nn
from typing import Dict, List
from copy import deepcopy

from .modules.global_area import EmptyGlobalVar
from .modules.encoder import SREncoder
from .modules.decoder import SRDecoder
from .modules.communicator import SRCommunicator
from .utils import HoldoutNeuron, HParams, SaveVariables, Batch, deep_clone_tensors
from .blocks import MLPBase

class MRLFADS(pl.LightningModule):
    """Multi-Regional Latent Factor Analysis via Dynamical Systems (MRLFADS).
    
    This module implements the MRLFADS model, a sequential autoencoder
    framework for jointly modeling neural population activity recorded
    from multiple brain regions. 
    
    The model is described in the corresponding OpenReview paper:
        https://openreview.net/forum?id=O14GjxDAt3
    """
    def __init__(
        self,
        # ----- main parameters ---------------------- #
        area_params: dict,                                    # contains area-specific parameters
        num_other_areas: int,                                 # number of other areas (total areas - 1)
        seq_len: int,                                         # sequence length of data
        ic_enc_seq_len: int,                                  # sequence length for initial condition encoder to work with

        # ----- regularization parameters ------------ #
        l2_start_epoch: int,                                  # L2 is applied on all RNNs
        l2_increase_epoch: int,
        l2_scale: float,
        kl_start_epoch_co: int,
        kl_increase_epoch_co: int,
        kl_start_epoch_com: int,
        kl_increase_epoch_com: int,
        kl_ic_scale: float,
        kl_co_scale: float,
        kl_com_scale: float,

        kl_start_epoch_gv: int = 0,
        kl_increase_epoch_gv: int = 1,
        kl_gv_scale: float = 0.0,
        kl_init_gv_scale: float = 0.0,
        l1_start_epoch_com: int = 0,                          # L1 is applied on communication channels
        l1_increase_epoch_com: int = 1,
        l1_com_scale: float = 0.0,

        # ----- default settings --------------------- #
        model_type: str = "rates",                            # communicates through rate/factor/generator
        hn_indices: dict = {},                                # heldout neuron indices
        kl_com_scale_override: dict = {},                     # kl_com specifications (if inhomogeneous across channels)
        recon_scale_override: dict = {},                      # recon ramping specifications (area specific)
        global_area=None,                                     # for global/external inputs
        detach_hn=True,                                       # detach holdout loss

        # ----- learning rate related parameters ----- #
        lr_init: float = 4.0e-3,
        lr_stop: float = 1.0e-5,
        lr_decay: float = 0.95,
        lr_patience: int = 12,
        lr_adam_beta1: float = 0.9,
        lr_adam_beta2: float = 0.999,
        lr_adam_epsilon: float = 1.0e-8,
        weight_decay: float = 0.0,
    ):
        """
        Args:
            area_params: Area-specific configuration (e.g., per-area architecture and priors).
                The expected keys and nested structure are determined by this codebase's
                single area model components.
            num_other_areas: Number of other areas (total areas minus one).
            seq_len: Length of the input sequences used for encoding and reconstruction.
            ic_enc_seq_len: Sequence length used by the initial condition encoder.

            l2_start_epoch: Epoch at which L2 regularization begins to apply.
            l2_increase_epoch: Number of epochs over which L2 regularization is ramped.
            l2_scale: Overall L2 regularization scale.

            kl_start_epoch_co: Epoch to start KL regularization for inferred inputs.
            kl_increase_epoch_co: Number of epochs over which KL for inferred inputs is ramped.
            kl_start_epoch_com: Epoch to start KL regularization for inferred messages.
            kl_increase_epoch_com: Number of epochs over which KL for inferred messages is ramped.
            kl_ic_scale: KL weight for inferred initial conditions.
            kl_co_scale: KL weight for inferred inputs.
            kl_com_scale: KL weight for inferred messages.

            kl_start_epoch_gv: Epoch to start KL regularization for global inputs. Defaults to 0.
            kl_increase_epoch_gv: Number of epochs over which KL for global inputs is
                ramped. Defaults to 1.
            kl_gv_scale: KL weight for global variables/inputs. Defaults to 0.0.

            l1_start_epoch_com: Epoch to begin applying L1 regularization to communication
                channels. Defaults to 0.
            l1_increase_epoch_com: Number of epochs over which L1 is ramped. Defaults to 1.
            l1_com_scale: L1 weight for communication channels. Defaults to 0.0.

            model_type: Communication mode used in the model. Defaults to "rates".
            hn_indices: Mapping that specifies held-out neuron indices. Defaults to an empty dict.
            kl_com_scale_override: Optional per-target-area overrides for message KL scaling 
                when the KL weight is not homogeneous. Defaults to an empty dict.
            recon_scale_override: Optional per-area overrides for reconstruction loss scaling or
                ramping behavior. Defaults to an empty dict.
            global_area: Optional specification for global inputs shared across areas.
                Defaults to None.

            lr_init: Initial learning rate. Defaults to 4e-3.
            lr_stop: Minimum learning rate after decay. Defaults to 1e-5.
            lr_decay: Multiplicative decay factor for the learning rate schedule. Defaults to 0.95.
            lr_patience: Number of epochs to wait before decaying the learning rate. Defaults to 12.
            lr_adam_beta1: Adam beta1 coefficient. Defaults to 0.9.
            lr_adam_beta2: Adam beta2 coefficient. Defaults to 0.999.
            lr_adam_epsilon: Adam epsilon for numerical stability. Defaults to 1e-8.
            weight_decay: Optimizer weight decay. Defaults to 0.0.
        """
        super().__init__()
        
        # ----- Hyperparameters -------------- #
        self.save_hyperparameters(
            ignore = ["area_params", "hn_indices", "global_area"]
        )
        self.area_params = area_params
        self.hparams.hn_indices = hn_indices

        # ----- Setups ----------------------- #
        # Build global area (defaults to placeholder `EmptyGlobalVar`)
        if isinstance(global_area, type(None)): self.global_area = EmptyGlobalVar(self.area_params)
        else: self.global_area = global_area
        self.hparams.gv_dim = self.global_area.hparams.gv_dim
        
        # Build all the areas (SR-LFADS)
        self.area_names = list(area_params.keys())
        self._build_areas(area_params)
            
        # Build heldout validation for pre- and post-processing
        self.holdout = HoldoutNeuron(self.hparams)

    def forward(
        self,
        batch: dict,
        sample: bool = False,
    ):
        """
        Args:
            batch: Mapping from session index to a `Batch` namedtuple containing the
                inputs for that session.
            sample: If True, sample stochastic latents. If False, use deterministic statistics.
                Defaults to False.

        Notes:
            Decoding follows the order of computation:

                s_{t-1, o} -> u_{t, s}, m_{t, s} -> f_{t, s} -> r_{t, s}

            where:
                - s: Encoded data.
                - r: Rates.
                - u: Inferred inputs.
                - m: Communication messages.
                - f: Latent factors.
                - First subscript index is time step t.
                - Second subscript index is area: `s` for self, `o` for other area.
        """
        hps = self.hparams
        
        # ----- Initial Setups --------------- #
        sessions = sorted(batch.keys()) # session structure is provided but not used currently
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        batch_size = sum(batch_sizes) # combine batches from all sessions
        self._build_save_var(batch_sizes) # build variables to store intermediate results
        self.global_area.build(self.current_info, self.current_batch) # build global area based on data/metadata
        
        # ----- Run ENCODE ------------------- #
        emission_state_dict = {}
        for ia, (area_name, area) in enumerate(self.areas.items()):
            
            ahps = area.hparams
            
            # readin --> encoder
            data = torch.cat([area.readin[s](batch[s].encod_data[area_name].float()) for s in sessions])
            (ic_mean, ic_std, ci), (con_init, gen_init, factor_init) = area.encoder(data.float(), sample=sample)
            factor_init_split = torch.split(factor_init, batch_sizes)
            rates_init = torch.cat([area.readout[s](factor_init_split[s]) for s in sessions])
            
            # Save the results
            state = torch.cat([torch.tile(con_init, (batch_size, 1)), gen_init, factor_init], dim=1)
            self.save_var[area_name].states[:,0,:] = state # this causes state to have +1 length
            self.save_var[area_name].inputs[..., :area.hparams.ci_size] = ci
            self.save_var[area_name].ic_params = torch.cat([ic_mean, ic_std], dim=1)
            
            # Save <...> as emission according to model type
            if hps.model_type == "rates":
                emission_state_dict[area_name] = rates_init
            elif hps.model_type == "factor":
                emission_state_dict[area_name] = factor_init
            elif hps.model_type == "generator":
                emission_state_dict[area_name] = gen_init
            else:
                raise ValueError(f"``model_type`` cannot be {hps.model_type}")
            
        # ----- Run DECODE ------------------- #
        for t in range(hps.seq_len - hps.ic_enc_seq_len):
            
            # Initialize new emission tensor dict to store rates/factors/etc. from each area
            emission_state_dict_new = {}
            
            for ia, (area_name, area) in enumerate(self.areas.items()):
                
                ahps = area.hparams
                
                # Communicator (used only if there are >1 areas)
                if area.communicator:
                    mask_hn = (len(self.hparams.hn_indices) > 0) # heldout neuron mask
                    com_samp, com_params = area.communicator(emission_state_dict, sample=sample, mask_hn=mask_hn)
                    self.save_var[area_name].inputs[:, t, ahps.ci_size: ahps.ci_size + ahps.com_dim * ahps.num_other_areas] = com_samp
                    self.save_var[area_name].com_params[:,t,:] = com_params
                    
                # If communicator is None, and if num_other_areas is 0, then it is single regional
                else:
                    pass
                
                # External input (from data, e.g. perturbation) and global variable (defined; inferred; etc.)
                gv_params, gv_samp = self.global_area(
                    area_name, t,
                    sample=sample,
                    kwargs={'emission_state_dict': emission_state_dict},
                ) # all areas get the same gv (+ sample difference)
                ext_input = torch.cat([batch[s].ext_input[area_name] for s in sessions])
                ext_input = torch.cat([ext_input[:,t], gv_samp.to(self.device)], dim=-1)
                self.save_var[area_name].gv_params[:,t,:] = gv_params
                self.save_var[area_name].ext_inputs[:,t,:] = ext_input
                
                # Decoder
                states = self.save_var[area_name].states[:,t,:].clone()
                inputs = self.save_var[area_name].inputs[:,t,:]
                inputs = torch.cat([inputs, ext_input], dim=1)
                new_state, co_params, con_samp = area.decoder(inputs, states, sample=sample)  
                self.save_var[area_name].states[:,t+1,:] = new_state
                self.save_var[area_name].co_params[:,t,:] = co_params
                if not isinstance(con_samp, type(None)):
                    self.save_var[area_name].inputs[:,t,-ahps.co_dim:] = con_samp
                
                # factor --> readout --> rates
                factor_state_split = torch.split(new_state[..., -ahps.fac_dim:], batch_sizes)
                for s in sessions:
                    rates = area.readout[s](factor_state_split[s])
                    self.outputs[area_name][s][:,t,:] = rates
                
                # Append to emission
                if hps.model_type == "rates":
                    emission_state = rates
                elif hps.model_type == "factor":
                    emission_state = new_state[..., -ahps.fac_dim:]
                elif hps.model_type == "generator":
                    emission_state = new_state[..., ahps.con_size:-ahps.fac_dim]
                else:
                    raise ValueError(f"``model_type`` cannot be {hps.model_type}")
                    
                emission_state_dict_new[area_name] = emission_state
                    
                # Process heldout neurons
                if len(self.hparams.hn_indices) > 0:
                    for s in sessions:
                        if hps.detach_hn: # so that it doesn't backprop through factor
                            preds = area.predictor[s](factor_state_split[s].clone().detach()) 
                        else:
                            preds = area.predictor[s](factor_state_split[s].clone())
                        self.preds[area_name][s][:,t,:] = preds
                
            # Reset emission based on activity of current time step
            emission_state_dict = emission_state_dict_new
                
        return self.outputs

    def _shared_step(self, batch, batch_idx, step_type):
        
        hps = self.hparams
        num_areas = len(self.areas)

        # ----- Heldout Neurons -------- #
        sessions = sorted(batch.keys())
        self.current_info = {s: b[1] for s, b in batch.items()} 
        batch = {s: b[0] for s, b in batch.items()} # ignore info, only data is relevant
        self.raw_batch = {
            0: Batch(
                encod_data=deep_clone_tensors(batch[0].encod_data),
                ext_input=deep_clone_tensors(batch[0].ext_input),
            )
        } # stores data before heldout neurons are masked
        
        batch = {s: self.holdout.preprocess(batch[s]) for s in sessions}
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        self.batch_size = batch_size = sum(batch_sizes)
        self.current_batch = batch # stores data that forward() step sees
        
        # ----- Forward pass ----------------- #
        self.forward(
            batch,
            sample = (step_type == "train"),
        )
        
        # ----- LOSS Calculation ------------- #
        # Compute ramping coefficients
        l1_ramp = self._compute_ramp(hps.l1_start_epoch_com, hps.l1_increase_epoch_com)
        l2_ramp = self._compute_ramp(hps.l2_start_epoch, hps.l2_increase_epoch)
        kl_ramp_u = self._compute_ramp(hps.kl_start_epoch_co, hps.kl_increase_epoch_co)
        kl_ramp_m = self._compute_ramp(hps.kl_start_epoch_com, hps.kl_increase_epoch_com)
        kl_ramp_g = self._compute_ramp(hps.kl_start_epoch_gv, hps.kl_increase_epoch_gv, init=hps.kl_init_gv_scale)
        
        # Calculate all losses
        mr_loss, mr_recon, mr_l2, mr_kl_u, mr_kl_m, mr_kl_g, mr_hn_loss = 0, 0, 0, 0, 0, 0, 0
        comm_norms = [] # for L1 loss
        
        for area_name, area in self.areas.items():
            
            ahps = area.hparams
            other_area_names = deepcopy(self.area_names)
            other_area_names.remove(area_name)
            
            # ===== Reconstruction Loss ========== #
            rates_split = self.outputs[area_name]
            recon_all = [
                area.output_dist(
                    batch[s].encod_data[area_name][:,hps.ic_enc_seq_len:],
                    rates_split[s])
                for s in sessions
            ]
            recon_all = [
                 self.holdout.postprocess_holdin(
                    recon_all[s],
                    area_name = area_name,
                )
            for s in sessions
            ]
            recon_all = [torch.sum(ra, dim=(1, 2)) for ra in recon_all] # uses sum for time and neuron dimension
            sess_recon = [ra.mean() for ra in recon_all]
            recon = torch.mean(torch.stack(sess_recon))
            mr_recon += recon
            
            # Reconstruction ramp: area specific
            if area_name in hps.recon_scale_override.keys():
                recon_start_epoch, recon_increase_epoch, recon_final_scale = hps.recon_scale_override[area_name]
                current_ramp = self._compute_ramp(recon_start_epoch, recon_increase_epoch) # from 0 -> 1
                recon_ramp = max(1 - current_ramp, recon_final_scale) # from 1 -> recon_final_scale
            else:
                recon_ramp = 1.0
            
            # ===== Heldout Neuron Loss ========== #
            if len(hps.hn_indices[area_name][0]) > 0: # no session implemented
                hn_loss = [
                    self.holdout.postprocess_holdout(
                        area_name,
                        self.preds[area_name][s],
                        area.output_dist,
                    )
                for s in sessions
                ]
                hn_loss = [torch.sum(hnl, dim=(1, 2)) for hnl in hn_loss]
                hn_loss= [hnl.mean() for hnl in hn_loss]
                hn_loss = torch.mean(torch.stack(hn_loss))
            else:
                hn_loss = 0.0
                
            mr_hn_loss += hn_loss
            
            # ===== L2 Loss ====================== #
            l2 = area.l2(hps.l2_scale)
            mr_l2 += l2
            
            # ===== KL Loss ====================== #
            # Initial condition
            ic_mean, ic_std = torch.split(self.save_var[area_name].ic_params, area.hparams.ic_dim, dim=1)
            ic_kl = area.ic_prior(ic_mean, ic_std) * area.hparams.kl_ic_scale
            
            # Inferred inputs
            u_kl = ic_kl
            if area.hparams.use_con:
                co_mean, co_std = torch.split(self.save_var[area_name].co_params, area.hparams.co_dim, dim=2)
                co_kl = area.co_prior(co_mean, co_std) * area.hparams.kl_co_scale
            else:
                co_kl = 0
            u_kl += co_kl
            mr_kl_u += u_kl
            
            # Communication
            if hps.num_other_areas > 0:
                com_mean, com_std = torch.split(
                    self.save_var[area_name].com_params,
                    area.hparams.com_dim * (num_areas-1),
                    dim=2,
                )
                
                if area_name not in hps.kl_com_scale_override.keys():
                    m_kl = area.com_prior(com_mean, com_std) * area.hparams.kl_com_scale
                    
                # Area-specific kl(m) penalties
                else:
                    # m_kl_seq is a list of length num_other_areas, with elements of shape (batch, time,)
                    m_kl_seq = area.com_prior(com_mean, com_std, reduction='batch', dim=ahps.com_dim) 
                    m_kls = []
                    for isrc, src in enumerate(other_area_names):
                        # Modify kl_com_scale if specified in override
                        if src in hps.kl_com_scale_override[area_name].keys():
                            kl_com_scale = hps.kl_com_scale_override[area_name][src]
                        else:
                            kl_com_scale = area.hparams.kl_com_scale
                          
                        # To mimic reduction='none', which is a (mean, sum, sum) operation across (batch, time, all channels), do:
                        # Step 1: sum over time dimension
                        m_kls.append( m_kl_seq[isrc].sum(dim=1, keepdim=True) * kl_com_scale ) # list with elements of shape (batch, 1)
                    # Step 2: sum over num_other_areas, because channels is num_other_areas * com_dim
                    m_kls = torch.cat(m_kls, dim=1).sum(dim=1) # shape = (batch,)
                    # Step 3: avg over batch dimension
                    m_kl = torch.mean(m_kls) # has been verified to be the same value as reduction='none'
                    
                mr_kl_m += m_kl
                
                # L1 on communication mean
                comm = com_mean.reshape(*com_mean.shape[:-1], hps.num_other_areas, -1) # shape = (batch, time, #areas, com_dim)
                comm_norms.append(torch.mean(comm.square(), dim=-1)) # shape = (batch, time, # upstream areas)
                
            else:
                m_kl = 0
                
            # Global variable
            if self.global_area.variational:
                gv_mean, gv_std = torch.split(self.save_var[area_name].gv_params, area.hparams.gv_dim, dim=2)
                g_kl = self.global_area.gv_prior(gv_mean, gv_std) * hps.kl_gv_scale
            else:
                g_kl = 0
            mr_kl_g += g_kl
                
            # ===== Final Loss =================== #
            sr_loss = recon_ramp * recon \
                        + l2_ramp * l2 \
                        + kl_ramp_u * u_kl \
                        + kl_ramp_m * m_kl \
                        + kl_ramp_g * g_kl \
                        + hn_loss
            mr_loss += sr_loss
            
            # Log area-speific information when on validation
            if step_type == "valid":
                area_metrics = {
                    f"{step_type}/{area_name}/recon": recon,
                    f"{step_type}/{area_name}/l2": l2,
                    f"{step_type}/{area_name}/kl/ic": ic_kl,
                    f"{step_type}/{area_name}/kl/co": co_kl,
                    f"{step_type}/{area_name}/kl/com": m_kl,
                    f"{step_type}/{area_name}/kl/gv": g_kl,
                    f"{step_type}/{area_name}/hn": hn_loss,
                }
                self.log_dict(
                    area_metrics,
                    on_step=False,
                    on_epoch=True,
                    batch_size=sum(batch_sizes),
                )
            
        # ===== L1 loss ====================== # (global sparsity)
        if len(comm_norms) > 0:
            comm_norms = torch.cat(comm_norms, dim=2) # shape = (batch, time, # connections)
            mr_l1 = torch.norm(comm_norms, 1, dim=-1).sum() / batch_size * hps.l1_com_scale
        else:
            mr_l1 = 0.0
        mr_loss += (l1_ramp * mr_l1)
            
        # ----- Log scalar metrics ----------- #
        metrics = {
            "cur_epoch": float(self.current_epoch),
            
            f"{step_type}/loss": mr_loss / num_areas,
            f"{step_type}/recon": mr_recon / num_areas,
            f"{step_type}/l2": mr_l2 / num_areas,
            f"{step_type}/l1": mr_l1 / num_areas,
            f"{step_type}/kl/u": mr_kl_u / num_areas,
            f"{step_type}/kl/m": mr_kl_m / num_areas,
            f"{step_type}/kl/g": mr_kl_g / num_areas,
            f"{step_type}/hn": mr_hn_loss / num_areas,
            
            f"{step_type}/l1/ramp": l1_ramp,
            f"{step_type}/l2/ramp": l2_ramp,
            f"{step_type}/kl/ramp/u": kl_ramp_u,
            f"{step_type}/kl/ramp/m": kl_ramp_m,
            f"{step_type}/kl/ramp/g": kl_ramp_g,
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=sum(batch_sizes),
        )
        
        # Allowing algorithm to not error out
        if torch.isnan(mr_loss): return None
        return mr_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
        
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "valid")
        
    def predict_step(self, batch, batch_idx, sample=False):
        return self._shared_step(batch, batch_idx, "valid")
    
    def configure_optimizers(self):
        hps = self.hparams
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(hps.lr_adam_beta1, hps.lr_adam_beta2),
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=hps.lr_decay,
            patience=hps.lr_patience,
            threshold=0.01,
            threshold_mode="rel",
            cooldown=1,
            min_lr=hps.lr_stop,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid/recon",
        }

    def _build_areas(self, area_params):
        # MR-LFADS hyperparameters to copy into SR-LFADS hyperparameters
        hps = self.hparams
        hps_to_copy = ["seq_len", "ic_enc_seq_len", "num_other_areas", "hn_indices", "gv_dim"]
        mr_hps_dict = {key: self.hparams[key] for key in hps_to_copy}
        
        # Get total factor dimension for communication
        if hps.model_type == "rates":
            total_ems_dim_dict = {area_name: area_kwargs["num_neurons"]["n0"] for area_name, area_kwargs in area_params.items()}
        elif hps.model_type == "factor":
            total_ems_dim_dict = {area_name: area_kwargs["fac_dim"] for area_name, area_kwargs in area_params.items()}
        elif hps.model_type == "generator":
            total_ems_dim_dict = {area_name: area_kwargs["gen_size"] for area_name, area_kwargs in area_params.items()}
        else:
            raise ValueError(f"``model_type`` cannot be {hps.model_type}")
        mr_hps_dict.update({"total_ems_dim_dict": total_ems_dim_dict,
                            "area_names": self.area_names})
        
        # Build all SR-LFADS instances
        self.areas = nn.ModuleDict()
        for area_name, area_kwargs in area_params.items():
            area_kwargs.update(mr_hps_dict)
            self.areas[area_name] = SRLFADS(area_name, **area_kwargs)
            
    def _build_save_var(self, batch_sizes):
        self.save_var = {}
        self.outputs = {}
        self.preds = {}
        
        batch_size = sum(batch_sizes)
        target_len = self.hparams.seq_len - self.hparams.ic_enc_seq_len
        num_other_areas = len(self.area_names) - 1 # number of other areas
        
        for area_name, area in self.areas.items():
            ahps = self.areas[area_name].hparams
            self.save_var[area_name] = SaveVariables(
                # states has 1 extra time in the beginning
                states = torch.zeros(batch_size, target_len+1, ahps.con_size + ahps.gen_size + ahps.fac_dim).to(self.device),
                inputs = torch.zeros(batch_size, target_len, ahps.ci_size + ahps.com_dim * num_other_areas + ahps.co_dim).to(self.device),
                ext_inputs = torch.zeros(batch_size, target_len, ahps.ext_input_dim + ahps.gv_dim).to(self.device),
                ic_params = torch.zeros(batch_size, 2 * ahps.ic_dim).to(self.device),
                co_params = torch.zeros(batch_size, target_len, 2 * ahps.co_dim).to(self.device),
                com_params = torch.zeros(batch_size, target_len, 2 * ahps.com_dim * num_other_areas).to(self.device), 
                gv_params = torch.zeros(batch_size, target_len, 2 * ahps.gv_dim).to(self.device), 
            )
            
            self.outputs[area_name] = []
            for i_sess in range(len(ahps.num_neurons)):
                self.outputs[area_name].append(
                torch.zeros(batch_sizes[i_sess], target_len, ahps.num_neurons[i_sess] * area.output_dist.n_params).to(self.device)
                )
                
            if len(self.hparams.hn_indices) > 0:
                self.preds[area_name] = []
                for i_sess in range(len(ahps.num_neurons)):
                    self.preds[area_name].append(
                    torch.zeros(batch_sizes[i_sess], target_len, area.n_heldout[i_sess] * area.output_dist.n_params).to(self.device)
                    )
        
    def _compute_ramp(self, start, increase, init=0.0):
        return self._compute_ramp_inner(self.current_epoch, start, increase, init=init)
    
    @staticmethod
    def _compute_ramp_inner(epoch, start, increase, init=0.0):
        # progress from 0 to 1
        progress = (epoch + 1 - start) / (increase + 1)
        progress = torch.clamp(torch.tensor(progress), 0.0, 1.0)

        # scale progress from init to 1
        ramp = init + (1.0 - init) * progress
        return ramp

class SRLFADS(nn.Module):
    """Single-Regional Latent Factors Analysis via Dynamical Systems (SRLFADS).
    
    The present implementation follows the model structure described in:
        Pandarinath et al., "Inferring single-trial neural population dynamics
        using sequential auto-encoders," Nature Methods, 2018.
        https://doi.org/10.1038/s41592-018-0109-9
    with modifications and extensions specific to this codebase.  
    """
    def __init__(
        self,
        area_name: str,
        ic_prior: nn.Module,
        co_prior: nn.Module,
        com_prior: nn.Module,
        readin: nn.ModuleList,
        readout: nn.ModuleList,
        output_dist: nn.ModuleList,
        **kwargs,
    ):
        """
        Args:
            area_name: Name or identifier of the brain area.
            ic_prior: Prior module for the initial condition latent variables.
            co_prior: Prior module for inferred inputs.
            com_prior: Prior module for communication messages.
            readin: Feedforward layers used to embed observed data.
            readout: Feedforward layers used to reconstruct observed data.
            output_dist: Modules defining the output probability distribution
                over neuronal activity.

        Keyword Args:
            num_neurons (int): Number of neurons in the area.
            data_dim (int): Dimensionality of the observed data.
            ext_input_dim (int): Dimensionality of external inputs.
            co_dim (int): Dimensionality of inferred inputs.
            ic_dim (int): Dimensionality of the initial condition latent state.
            fac_dim (int): Dimensionality of latent factors.
            com_dim (int): Dimensionality of communication messages.

            ic_enc_size (int): Hidden size of the bidirectional GRU used for
                encoding the initial condition.
            ci_size (int): Hidden size of the unidirectional GRU used to encode
                inputs to the controller.
            con_size (int): Hidden size of the controller network.
            gen_size (int): Hidden size of the generator network.

            seq_len (int): Length of the input data sequences (trial duration).
            ic_enc_seq_len (int): Number of time steps used to infer the initial
                condition. Must be greater than zero.

            kl_ic_scale (float): KL divergence weight for initial condition
                latents.
            kl_co_scale (float): KL divergence weight for inferred inputs.
            kl_com_scale (float): KL divergence weight for communication messages.
            kl_gv_scale (float): KL divergence weight for global variables.
        """
        super().__init__()
        
        # Initialize hyperparameters passed down from MRLFADS
        hparam_keys = [
            "total_fac_dim", "data_dim", "seq_len", "ext_input_dim", "ic_enc_seq_len",
            "ic_enc_size", "ci_size", "con_size", "co_dim", "ic_dim", "gen_size", "fac_dim", "gv_dim",
            "com_dim", "num_neurons", "kl_ic_scale", "kl_co_scale", "kl_com_scale", "kl_gv_scale",
        ]
        hparam_dict = {key: None for key in hparam_keys}
        hparam_dict.update(kwargs)
        
        # Setup HParam
        hparam_dict["num_neurons"] = list(hparam_dict["num_neurons"].values())
        hps = self.hparams = HParams(hparam_dict)
        self.hparams.add("co_prior", co_prior)
        self.name = area_name

        # Set up model components
        hps.use_con = all([self.hparams.ci_size > 0, self.hparams.con_size > 0, self.hparams.co_dim > 0])
        self.readin = readin
        self.encoder = SREncoder(self.hparams, ic_prior)
        self.decoder = SRDecoder(self.hparams)
        self.readout = readout
        self.output_dist = output_dist
        self.ic_prior = ic_prior
        self.co_prior = co_prior
        self.com_prior = com_prior
        
        # Set up communicator
        if self.hparams.num_other_areas != 0:
            self.communicator = SRCommunicator(hps, com_prior, area_name)
        else:
            self.communicator = None
            
        # Set up linear layer for predicting heldout neuron activity from heldin latent factors
        if len(hps.hn_indices) > 0:
            self.predictor = nn.ModuleList()
            self.n_heldout = []
            for s in range(len(readin)):
                n_heldout = len(hps.hn_indices[self.name][s])
                self.predictor.append( MLPBase([[hps.fac_dim, n_heldout * self.output_dist.n_params, None]]) )
                self.n_heldout.append(n_heldout)
    
    def forward(self): raise NotImplementedError("Forward() function should not be called in SRLFADS.")
    
    def l2(self, scale):
        weights = [
            self.decoder.gen_cell.weight_hh,
            self.encoder.ic_enc.fwd.inner.weight_hh,
            self.encoder.ic_enc.bwd.inner.weight_hh,
        ]

        if self.hparams.use_con:
            weights += [
                self.encoder.ci_enc.inner.weight_hh,
                self.encoder.ci_enc.inner.weight_hh,
                self.decoder.con_cell.weight_hh,
            ]
            
        l2 = 0.0
        total_elems = 0
        for w in weights:
            l2 += scale * 0.5 * torch.norm(w, 2) ** 2
            total_elems += w.numel()
        return l2 / (total_elems + 1e-8)

