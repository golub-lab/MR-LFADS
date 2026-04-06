import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from .blocks.encoder import SREncoder, BiEncoder
from .blocks.decoder import SRDecoder
from .blocks.communicator import Communicator
from .blocks.globalvars import EmptyGlobalVar
from .blocks.processors import CommunicatorProcessor, DecoderProcessor, ReadoutProcessor
from .utils.common_utils import Batch, SaveVariables, HoldoutNeuron, HParams, pad_by_index, deep_clone_tensors, det
from .utils.torch_utils import MLPBase, EMAMetric

# If TensorCores are available
torch.set_float32_matmul_precision('high')

class MRLFADS(pl.LightningModule):
    """
    Multi-Regional Latent Factor Analysis via Dynamical Systems (MRLFADS).
    
    This module implements the MRLFADS model, a sequential autoencoder
    framework for jointly modeling neural population activity recorded
    from multiple brain regions. 
    """
    def __init__(
        self,
        # ----- main parameters ---------------------- #
        areas_params: dict,                                   # contains area-specific parameters
        num_other_areas: int,                                 # number of other areas (total areas - 1)
        seq_len: int,                                         # sequence length of data
        ic_enc_seq_len: int,                                  # sequence length for i.c. encoder
        
        # ----- regularization parameters ------------ #
        l2_start_epoch: int,
        l2_increase_epoch: int,
        l2_scale: float,
        kl_start_epoch_co: int,
        kl_increase_epoch_co: int,
        kl_start_epoch_com: int,
        kl_increase_epoch_com: int,
        kl_ic_scale: float,
        kl_co_scale: float,
        kl_com_scale: float,
        
        kl_co_scale_init: float = 0.0,
        kl_start_epoch_gv: int = 0,
        kl_increase_epoch_gv: int = 1,
        kl_gv_scale: float = 0.0,
        
        # ----- learning rate related parameters ----- #
        lr_scheduler_type: str = 'ReduceLROnPlateau',
        lr_init: float = 4.0e-3,
        lr_stop: float = 1.0e-5,
        lr_decay: float = 0.95,
        lr_patience: int = 6,
        lr_adam_beta1: float = 0.9,
        lr_adam_beta2: float = 0.99,
        lr_adam_epsilon: float = 1.0e-8,
        weight_decay: float = 0.0,
        
        # ----- misc parameters ---------------------- #
        dropout_rate: float = 0.3,
        cell_clip: float = 5.0,
        hn_indices: dict = {},                                # holdout neuron indices
        kl_com_scale_override: dict = {},                     # kl_com specifications
        global_area = None,
        detach_hn = True,
    ):
        """
        Args:
            areas_params: A dictionary where keys are area names and values are dictionaries of 
                area-specific parameters (e.g., number of neurons, readin/readout modules, priors, etc.).
            num_other_areas: The number of other areas (total areas - 1) used for communication.
            seq_len: The length of the input data sequences (trial duration).
            ic_enc_seq_len: The number of time steps used to infer the initial condition. 
                Must be greater than zero.

            l2_start_epoch: The epoch at which to start applying L2 regularization.
            l2_increase_epoch: The number of epochs over which to increase L2 regularization to its 
                full value.
            l2_scale: The coefficient of the L2 regularization term.

            kl_start_epoch_co: The epoch at which to start applying KL regularization for inferred inputs.
            kl_increase_epoch_co: The number of epochs over which to increase KL regularization for 
                inferred inputs to its full value.
            kl_start_epoch_com: The epoch at which to start applying KL regularization for communication 
                messages.
            kl_increase_epoch_com: The number of epochs over which to increase KL regularization for 
                communication messages to its full value.
            kl_ic_scale: The coefficient of the KL regularization term for initial condition latents.
            kl_co_scale: The coefficient of the KL regularization term for inferred inputs.
            kl_com_scale: The coefficient of the KL regularization term for communication messages.
            
            kl_co_scale_init: The initial coefficient of the KL regularization term for inferred inputs at the start of training (before ramping up).
            kl_start_epoch_gv: The epoch at which to start applying KL regularization for global variables.
            kl_increase_epoch_gv: The number of epochs over which to increase KL regularization for global variables to its full value.
            kl_gv_scale: The coefficient of the KL regularization term for global variables.
            
            lr_scheduler_type: The type of learning rate scheduler to use (currently, only 
                'ReduceLROnPlateau' is supported).
            lr_init: The initial learning rate for the optimizer.
            lr_stop: The minimum learning rate for the 'ReduceLROnPlateau' scheduler.
            lr_decay: The factor by which to reduce the learning rate for the 'ReduceLROnPlateau' 
                scheduler.
            lr_patience: The number of epochs with no improvement after which to reduce the learning rate 
                for the 'ReduceLROnPlateau' scheduler.
            lr_adam_beta1: The beta1 parameter for the Adam optimizer.
            lr_adam_beta2: The beta2 parameter for the Adam optimizer.
            lr_adam_epsilon: The epsilon parameter for the Adam optimizer.
            weight_decay: The weight decay (L2 regularization) parameter for the optimizer.

            dropout_rate: The dropout rate to use in the model.
            cell_clip: The value to which to clip the hidden states of RNN cells.

            hn_indices: A dictionary specifying the indices of heldout neurons for each area and session.
            kl_com_scale_override: A dictionary specifying any overrides for the KL divergence weight for 
                communication messages, keyed by area name.
            global_area: An optional module defining the global area. If None, a placeholder empty global area 
                will be used.
            detach_hn: Whether to detach the factor states when computing heldout neuron predictions 
                (to prevent gradients from flowing through them).
        """
        super().__init__()
        
        # ----- Hyperparameters -------------- #
        
        self.save_hyperparameters(
            ignore = ["global_area"]
        )
        hps = self.hparams  
        self.fit_metric = EMAMetric(momentum=0.3) 

        # ----- Other Setups ----------------- #
        
        # Build global area (defaults to placeholder `EmptyGlobalVar`)
        if isinstance(global_area, type(None)): self.global_area = EmptyGlobalVar(hps.areas_params)
        else: self.global_area = global_area
        hps.gv_dim = self.global_area.hparams.gv_dim
        
        # Build all the areas (SR-LFADS)
        self.area_names = list(areas_params.keys())
        self._build_areas(areas_params)
            
        # Build heldout validation for pre- and post-processing
        self.holdout = HoldoutNeuron(self.hparams)
        
        # Build parallel processors (for faster training)
        cprocessor = CommunicatorProcessor(self.areas)
        self.cprocessor = torch.compile(
            cprocessor,
            mode="default",  
            fullgraph=False,
        )
        dprocessor = DecoderProcessor(self.areas)
        self.dprocessor = torch.compile(
            dprocessor,
            mode="default",
            fullgraph=False,
        )
        rprocessor = ReadoutProcessor(self.areas)
        self.rprocessor = torch.compile(
            rprocessor,
            mode="default",
            fullgraph=False,
        )

    def forward(
        self,
        batch: dict,
        sample: bool = False,
    ):
        hps = self.hparams
        
        # ----- Initial Setups --------------- #
        # Calculate total batch_size
        sessions = sorted(batch.keys())
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        batch_size = sum(batch_sizes)
        self._build_save_var(batch_sizes)
        
        # Concat external inputs
        ext_inputs = []
        for ia, (area_name, area) in enumerate(self.areas.items()):
            ahps = area.hparams
            ext_input = torch.cat([batch[s].ext_input[area_name] for s in sessions])
            ext_inputs.append(ext_input.float())
        ext_inputs = tuple(ext_inputs)
        
        # ----- Run ENCODE ------------------- #
        emission_states = []
        area_data_dict = {} # store data after readin
        for ia, (area_name, area) in enumerate(self.areas.items()):
            
            ahps = area.hparams
            
            # readin --> encoder --> icsampler (controller, generator and factors)
            data = torch.cat([area.readin[s](batch[s].encod_data[area_name].float()) for s in sessions])
            area_data_dict[area_name] = data
            (ic_params, ci), (con_init, gen_init, factor_init) = area.encoder(
                data.float(), sample=sample
            )
            factor_init_split = torch.split(factor_init, batch_sizes)
            rates_init = [area.readout[s](factor_init_split[s]) for s in sessions]
            
            # Save the results
            state = torch.cat([torch.tile(con_init, (batch_size, 1)), gen_init, factor_init], dim=1)
            self.save_var[area_name].states[:,0,:] = state # this causes state to have +1 length
            self.save_var[area_name].inputs[..., :area.hparams.ci_enc_dim] = ci
            self.save_var[area_name].ic_params = ic_params
            
            # Save <...> as emission according to model type
            rates_init = pad_by_index(rates_init, area.hparams.num_neurons, area.output_dist.n_params)
            emission_states.append( rates_init )
                
        self.global_area.build(
            self.current_info,
            self.current_batch,
            area_data_dict,
            hps,
        ) # build global area based on data/metadata
            
        # ----- Run DECODE ------------------- #
        for t in range(hps.seq_len - hps.ic_enc_seq_len):
            
            # ----- Communicator ----------------- # 
            # Build noise for communicator sampling
            com_noise = []
            for ia, (area_name, area) in enumerate(self.areas.items()):
                ahps = area.hparams
                eps = torch.randn(batch_size, ahps.com_dim * hps.num_other_areas, device=data.device)
                com_noise.append(eps)
            
            # Communicator step (parallel)
            com_outputs, com_params = self.cprocessor(
                emission_states,
                sample=sample,
                mask_hn=(len(self.hparams.hn_indices) > 0),
                com_noise = tuple(com_noise),
            )
            
            # Store communicator outputs for all areas
            for ia, area_name in enumerate(self.area_names):
                ahps = self.areas[area_name].hparams
                if com_outputs[ia] is not None:
                    self.save_var[area_name].inputs[:, t, ahps.ci_enc_dim: ahps.ci_enc_dim + ahps.com_dim * ahps.num_other_areas] = com_outputs[ia]
                    self.save_var[area_name].com_params[:, t, :] = com_params[ia]
               
            # ----- External Inputs -------------- # 
            # Global area input
            gv_params, gv_samp = self.global_area(
                t + hps.ic_enc_seq_len, # time is adjusted here
                sample=sample,
                emission_states=emission_states,
            )
            
            # Manual + global external input (sequential)
            ext_inputs_dict = {}
            for ia, area_name in enumerate(self.area_names):
                ext_input = torch.cat([ext_inputs[ia][:, t], gv_samp], dim=-1)
                self.save_var[area_name].gv_params[:,t,:] = gv_params
                self.save_var[area_name].ext_inputs[:,t,:] = ext_input.detach()
                ext_inputs_dict[area_name] = ext_input
                
            # ----- Decoder ---------------------- # 
            # Prepare inputs for all areas' decoders
            decoder_inputs = []
            decoder_states = []

            for area_name in self.area_names:
                ahps = self.areas[area_name].hparams
                states = self.save_var[area_name].states[:, t, :]
                inputs = self.save_var[area_name].inputs[:, t, :]
                ext_input = ext_inputs_dict[area_name]
                inputs = torch.cat([inputs, ext_input], dim=1)

                decoder_inputs.append( inputs )
                decoder_states.append( states )
                
            # Decoder noise
            dec_noise = []
            for ia, (area_name, area) in enumerate(self.areas.items()):
                ahps = area.hparams
                eps = torch.randn(batch_size, ahps.co_dim, device=data.device)
                dec_noise.append(eps)
                
            # Decoder step (parallel)
            new_states, co_params, con_outputs = self.dprocessor(
                tuple(decoder_inputs),
                tuple(decoder_states),
                sample=sample,
                dec_noise=tuple(dec_noise),
            )
            
            # Store decoder outputs
            for ia, area_name in enumerate(self.area_names):
                ahps = self.areas[area_name].hparams
                new_state = new_states[ia]
                self.save_var[area_name].states[:, t+1, :] = new_state
                self.save_var[area_name].co_params[:, t, :] = co_params[ia]
                if con_outputs[ia] is not None:
                    self.save_var[area_name].inputs[:, t, -ahps.co_dim:] = con_outputs[ia]
                    
            # ----- Readout ---------------------- # 
            outputs_tuple = self.rprocessor(
                new_states,
                batch_sizes,
            )
            
            # Store readout outputs and compute emissions
            emission_states = []
            for ia, (area_name, area) in enumerate(self.areas.items()):
                ahps = self.areas[area_name].hparams
                rates_cat = []
                for s in sessions:
                    self.outputs[area_name][s][:, t, :] = outputs_tuple[ia][s]
                    rates_cat.append( outputs_tuple[ia][s] )

                # Append to emission for next timestep
                rates_cat = pad_by_index(rates_cat, area.hparams.num_neurons, area.output_dist.n_params)
                emission_states.append( rates_cat )
            
            emission_states = tuple(emission_states)
                
            # ----- Heldout ------------------- # 
            if len(self.hparams.hn_indices) > 0:
                for ia, area_name in enumerate(self.area_names):
                    ahps = self.areas[area_name].hparams
                    new_state = new_states[ia]
                    factor_state_split = torch.split(new_state[..., -ahps.fac_dim:], batch_sizes)

                    for s in sessions:
                        if hps.detach_hn:
                            preds = self.areas[area_name].predictor[s](
                                factor_state_split[s].clone().detach()
                            )
                        else:
                            preds = self.areas[area_name].predictor[s](factor_state_split[s])
                        self.preds[area_name][s][:, t, :] = preds
                
        return self.outputs

    def _shared_step(self, batch, batch_idx, step_type):
        
        hps = self.hparams
        num_areas = len(self.areas)

        # ----- Heldout Neurons -------- #
        sessions = sorted(batch.keys())
        self.current_info = {s: b[1] for s, b in batch.items()} 
        batch = {s: b[0] for s, b in batch.items()} # ignore info, only data is relevant
        self.raw_batch = {
            s: Batch(
                encod_data=deep_clone_tensors(batch[s].encod_data),
                ext_input=deep_clone_tensors(batch[s].ext_input),
            )
            for s in sessions
        } # stores data before heldout neurons are masked
        
        batch = {s: self.holdout.mask_data(batch[s], s) for s in sessions}
        batch_sizes = [batch[s].encod_data[self.area_names[0]].size(0) for s in sessions]
        self.batch_size = batch_size = sum(batch_sizes)
        self.current_batch = batch
        
        # ----- Forward pass ----------------- #
        self.forward(
            batch,
            sample = (step_type == "train"),
        )
        
        # ----- LOSS Calculation ------------- #
        # Compute ramping coefficients
        l2_ramp = self._compute_ramp(hps.l2_start_epoch, hps.l2_increase_epoch) # l1 shares this ramp
        kl_ramp_u = self._compute_ramp(hps.kl_start_epoch_co, hps.kl_increase_epoch_co, init=hps.kl_co_scale_init)
        kl_ramp_m = self._compute_ramp(hps.kl_start_epoch_com, hps.kl_increase_epoch_com)
        kl_ramp_g = self._compute_ramp(hps.kl_start_epoch_gv, hps.kl_increase_epoch_gv)
        
        # Calculate all losses
        mr_loss, mr_recon, mr_hn_loss = 0, 0, 0
        mr_l2, mr_kl_u, mr_kl_m, mr_kl_g, mr_r2 = 0, 0, 0, 0, 0
        
        for area_name, area in self.areas.items():
            
            ahps = area.hparams
            
            # ===== Reconstruction Loss ========== #
            rates_split = self.outputs[area_name]
            recon_all = [area.output_dist(
                    batch[s].encod_data[area_name][:,hps.ic_enc_seq_len:],
                    rates_split[s]
                )
                for s in sessions
            ]
            recon_all = [self.holdout.mask_holdout(
                recon_all[s],
                area_name,
                s,
            )
            for s in sessions]
            
            recon_all = [torch.sum(ra, dim=(1, 2)) for ra in recon_all] # uses sum except batch dim
            sess_recon = [ra.mean() for ra in recon_all]
            recon = torch.mean(torch.stack(sess_recon))
            mr_recon += recon
            
            # ===== Holdout Neuron Loss ========== #
            if len(hps.hn_indices) > 0:
                hn_loss = [self.holdout.compute_holdout(
                    area_name,
                    self.preds[area_name][s],
                    area.output_dist,
                    s)
                for s in sessions]
                hn_loss = [torch.sum(hnl, dim=(1, 2)) for hnl in hn_loss] # uses sum, not mean (except batch dim)
                hn_loss= [hnl.mean() for hnl in hn_loss]
                hn_loss = torch.mean(torch.stack(hn_loss))
            else:
                hn_loss = 0.0
                
            mr_hn_loss += hn_loss
            
            # ===== R-Square ===================== #
            r2_all = [area.output_dist.pseudo_r2(
                    batch[s].encod_data[area_name][:,hps.ic_enc_seq_len:],
                    rates_split[s])
                for s in sessions
            ]
            r2 = np.mean(r2_all)
            
            # ===== L2 Loss ====================== #
            l2 = area.l2()
            mr_l2 += l2
            
            # ===== KL Loss ====================== #
            ic_mean, ic_std = torch.split(self.save_var[area_name].ic_params, area.hparams.ic_dim, dim=1)
            ic_kl = area.ic_prior(ic_mean, ic_std) * area.hparams.kl_ic_scale
            
            u_kl = ic_kl
            if area.hparams.use_con:
                co_mean, co_std = torch.split(self.save_var[area_name].co_params, area.hparams.co_dim, dim=2)
                co_kl = area.co_prior(co_mean, co_std) * area.hparams.kl_co_scale
                u_kl = u_kl + co_kl
            else:
                co_kl = 0
            mr_kl_u += u_kl
            
            if hps.num_other_areas > 0:
                com_mean, com_std = torch.split(self.save_var[area_name].com_params, area.hparams.com_dim * (num_areas-1), dim=2)
                if hps.kl_com_scale_override == {}:
                    com_kl = area.com_prior(com_mean, com_std) * area.hparams.kl_com_scale
                    mr_kl_m += com_kl
                else:
                    com_kl = area.com_prior.kl_divergence_by_component(com_mean, com_std, 1, tpe="seq")
                
            else:
                com_kl = 0
            
            # Global area, only include if it is variational
            if self.global_area.variational:
                gv_mean, gv_std = torch.split(self.save_var[area_name].gv_params, area.hparams.gv_dim, dim=2)
                gv_kl = self.global_area.gv_prior(gv_mean, gv_std) * hps.kl_gv_scale
            else:
                gv_kl = 0
            mr_kl_g += gv_kl
                
            # ===== Final Loss =================== #
            sr_loss = recon\
                    + l2_ramp * l2\
                    + kl_ramp_u * u_kl\
                    + kl_ramp_m * com_kl\
                    + kl_ramp_g * gv_kl\
                    + hn_loss
            mr_loss += sr_loss
            mr_r2 += r2
            
            # Log area-speific information when on validation
            if step_type == "valid":
                area_metrics = {
                    f"{step_type}/{area_name}/recon": recon.detach(),
                    f"{step_type}/{area_name}/l2": l2.detach(),
                    f"{step_type}/{area_name}/kl/ic": ic_kl.detach(),
                    f"{step_type}/{area_name}/kl/co": det(co_kl),
                    f"{step_type}/{area_name}/kl/com": det(com_kl),
                    f"{step_type}/{area_name}/kl/gv": det(gv_kl),
                    f"{step_type}/{area_name}/r2": det(r2),
                    f"{step_type}/{area_name}/hn": det(hn_loss),
                }
                self.log_dict(
                    area_metrics,
                    on_step=False,
                    on_epoch=True,
                    batch_size=sum(batch_sizes),
                )
            
        # ----- Log scalar metrics ----------- #
        metrics = {
            f"{step_type}/loss": mr_loss.detach() / num_areas,
            f"{step_type}/recon": mr_recon.detach() / num_areas,
            f"{step_type}/l2": mr_l2.detach() / num_areas,
            f"{step_type}/kl/u": det(mr_kl_u) / num_areas,
            f"{step_type}/kl/m": det(mr_kl_m) / num_areas,
            f"{step_type}/kl/g": det(mr_kl_g) / num_areas,
            f"{step_type}/r2": det(mr_r2) / num_areas,
            f"{step_type}/hn": det(mr_hn_loss) / num_areas,
            
            f"{step_type}/l2/ramp": l2_ramp,
            f"{step_type}/kl/ramp/u": kl_ramp_u,
            f"{step_type}/kl/ramp/m": kl_ramp_m,
        }
        
        if step_type == "valid":
            self.fit_metric.update(mr_recon / num_areas, batch_size)
            metrics.update(
                {
                    "valid/recon_ema": self.fit_metric,
                    "hp_metric": recon,
                    "cur_epoch": float(self.current_epoch),
                }
            )
            
        if step_type != "predict":
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
        return self._shared_step(batch, batch_idx, "predict")
    
    def configure_optimizers(self):
        hps = self.hparams
        
        # Create an optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(hps.lr_adam_beta1, hps.lr_adam_beta2),
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        
        # Create a scheduler to reduce the learning rate over time
        if hps.lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=hps.lr_decay,
                patience=hps.lr_patience,
                threshold=0.0,
                min_lr=hps.lr_stop,
                verbose=True,
            )
        else:
            raise NotImplementedError(f"Unsupported lr_scheduler_type: {hps.lr_scheduler_type}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid/recon_ema",
        }

    def _build_areas(self, areas_params):
        # MR hyperparameters to copy into SR hyperparameters
        hps = self.hparams
        hps_to_copy = ["seq_len", "ic_enc_seq_len","num_other_areas", "hn_indices", 
                      "dropout_rate", "cell_clip", "l2_scale", "gv_dim"]
        mr_hps_dict = {key: self.hparams[key] for key in hps_to_copy}
        
        # Other area specific hyperparameters to copy into SR-LFADS hyperparameters
        other_area_hps = ['ci_enc_dim', 'data_dim', 'num_neurons']
        other_area_hps_dict = {
            area_name: {k: area_kwargs[k] for k in other_area_hps}
            for area_name, area_kwargs in areas_params.items()
        }
        mr_hps_dict.update({"other_area_hps_dict": other_area_hps_dict})
        
        # Get total factor dimension for communication
        total_ems_dim_dict = {area_name: sum(area_kwargs["num_neurons"].values()) for area_name, area_kwargs in areas_params.items()}
        mr_hps_dict.update({"total_ems_dim_dict": total_ems_dim_dict,
                            "area_names": self.area_names})
        
        # Build all SR-LFADS instances
        self.areas = nn.ModuleDict()
        for area_name, area_kwargs in areas_params.items():
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
                states = torch.empty(batch_size, target_len+1, ahps.con_dim + ahps.gen_dim + ahps.fac_dim, device=self.device),
                inputs = torch.empty(batch_size, target_len, ahps.ci_enc_dim + ahps.com_dim * num_other_areas + ahps.co_dim, device=self.device),
                ext_inputs = torch.empty(batch_size, target_len, ahps.ext_input_dim + ahps.gv_dim, device=self.device),
                ic_params = torch.empty(batch_size, 2 * ahps.ic_dim, device=self.device),
                co_params = torch.empty(batch_size, target_len, 2 * ahps.co_dim, device=self.device),
                com_params = torch.empty(batch_size, target_len, 2 * ahps.com_dim * num_other_areas, device=self.device),  
                gv_params = torch.empty(batch_size, target_len, 2 * ahps.gv_dim, device=self.device),
            )
            
            self.outputs[area_name] = []
            for i_sess in range(len(ahps.num_neurons)):
                self.outputs[area_name].append(
                torch.empty(batch_sizes[i_sess], target_len, ahps.num_neurons[i_sess] * area.output_dist.n_params, device=self.device)
                )
                
            if len(self.hparams.hn_indices) > 0:
                self.preds[area_name] = []
                for i_sess in range(len(ahps.num_neurons)):
                    self.preds[area_name].append(
                    torch.empty(batch_sizes[i_sess], target_len, area.n_heldout[i_sess] * area.output_dist.n_params, device=self.device)
                    )
        
    def _compute_ramp(self, start, increase, init=0.0):
        return self.compute_ramp_inner(self.current_epoch, start, increase, init=init)
    
    @staticmethod
    def compute_ramp_inner(epoch, start, increase, init=0.0):
        # Compute base ramp from 0 → 1
        ramp = (epoch + 1 - start) / (increase + 1)
        ramp = torch.clamp(torch.tensor(ramp), 0.0, 1.0)

        # Scale ramp so it starts at `init`
        return init + (1.0 - init) * ramp

class SRLFADS(nn.Module):
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

            ic_enc_dim (int): Hidden size of the bidirectional GRU used for
                encoding the initial condition.
            ci_enc_dim (int): Hidden size of the unidirectional GRU used to encode
                inputs to the controller.
            con_dim (int): Hidden size of the controller network.
            gen_dim (int): Hidden size of the generator network.

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
        
        hparam_keys = ["total_fac_dim", "data_dim", "seq_len", "ic_enc_seq_len", "ext_input_dim", 
                       "ic_enc_dim", "ci_enc_dim", "ci_lag", "con_dim", "gen_dim", "fac_dim",
                       "ic_dim", "co_dim", "com_dim", "gv_dim", 
                       "dropout_rate", "cell_clip", "num_neurons", "other_area_hps_dict",
                       "ic_post_var_min", "co_post_var_min", "com_post_var_min", 
                       "kl_ic_scale", "kl_co_scale", "kl_com_scale", "l2_scale", "ci_enc_type"]
        hparam_dict = {key: None for key in hparam_keys}
        hparam_dict["ci_enc_type"] = "unidirectional"
        hparam_dict.update(kwargs)
        
        hparam_dict["num_neurons"] = list(hparam_dict["num_neurons"].values())
        hps = self.hparams = HParams(hparam_dict)
        self.hparams.add("co_prior", co_prior)
        self.name = area_name
        assert len(readin) == len(readout)

        # Set up model components
        hps.use_con = all([self.hparams.ci_enc_dim > 0, self.hparams.con_dim > 0, self.hparams.co_dim > 0])
        self.readin = readin
        if hps.ci_enc_type == "unidirectional":
            self.encoder = SREncoder(self.hparams, ic_prior)
        elif hps.ci_enc_type == "bidirectional":
            assert hps.ci_enc_dim % 2 == 0 # ensure divisible by 2
            self.encoder = BiEncoder(self.hparams, ic_prior)
        else:
            raise ValueError()
        self.decoder = SRDecoder(self.hparams)
        self.readout = readout
        self.output_dist = output_dist
        self.ic_prior = ic_prior
        self.co_prior = co_prior
        self.com_prior = com_prior
        
        if self.hparams.num_other_areas != 0:
            self.communicator = Communicator(hps, com_prior, area_name)
        else:
            self.communicator = None
            
        if len(hps.hn_indices) > 0:
            self.predictor = nn.ModuleList()
            self.n_heldout = []
            for s in range(len(readin)):
                n_heldout = len(hps.hn_indices[self.name][s])
                self.predictor.append( MLPBase([[hps.fac_dim, n_heldout * self.output_dist.n_params, None]]) )
                self.n_heldout.append(n_heldout)
                
        self.registered_funcs = {}
    
    def forward(self): raise NotImplementedError
    
    def l2(self):
        hps = self.hparams
        recurrent_kernels_and_weights = [
            (self.decoder.gen_cell.weight_hh, hps.l2_scale),
            (self.encoder.ic_enc.fwd_gru.cell.weight_hh, hps.l2_scale),
            (self.encoder.ic_enc.bwd_gru.cell.weight_hh, hps.l2_scale),
        ]

        if hps.co_dim > 0:
            if hps.ci_enc_type == "unidirectional":
                recurrent_kernels_and_weights.extend(
                    [
                        (self.encoder.ci_enc.cell.weight_hh, hps.l2_scale),
                        (self.decoder.con_cell.weight_hh, hps.l2_scale),
                    ]
                )
            else:
                recurrent_kernels_and_weights.extend(
                    [
                        (self.encoder.ci_enc.fwd_gru.cell.weight_hh, hps.l2_scale),
                        (self.encoder.ci_enc.bwd_gru.cell.weight_hh, hps.l2_scale),
                        (self.decoder.con_cell.weight_hh, hps.l2_scale),
                    ]
                )
            
        recurrent_penalty = 0.0
        recurrent_size = 0
        for kernel, weight in recurrent_kernels_and_weights:
            if weight > 0:
                recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
                recurrent_size += kernel.numel()
        recurrent_penalty /= recurrent_size + 1e-8
        return recurrent_penalty