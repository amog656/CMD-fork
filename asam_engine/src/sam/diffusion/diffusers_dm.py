from typing import List, Tuple, Optional, Callable, Union
from contextlib import contextmanager, nullcontext
import types
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
# try:
#     from diffusers import EDMEulerScheduler
# except ImportError:
#     import diffusers
#     EDMEulerScheduler = None
from sam.diffusion.common import DiffusionCommon


class Diffusers(DiffusionCommon):

    def __init__(self,
                 eps_model: Callable,
                 sched_params: dict,
                 loss: str = "l2",
                 extra_loss: dict = {},
                 ema=None,
                 sc_params=None):

        # Setup the network and other core parameters.
        self.eps_model = eps_model
        self.ema = ema
        if not loss in ("l2",):  # ("l1", "l2", "huber"):
            raise KeyError(loss)
        self.loss_type = loss

        # Setup the diffusers scheduler.
        if sched_params["name"] == "ddpm":
            # From: https://huggingface.co/docs/diffusers/api/schedulers/ddpm
            #       https://huggingface.co/docs/diffusers/api/pipelines/ddpm
            self.sched = DDPMScheduler(
                num_train_timesteps=sched_params["num_train_timesteps"],
                beta_start=sched_params["beta_start"],
                beta_end=sched_params["beta_end"],
                beta_schedule=sched_params["beta_schedule"],
                trained_betas=None,
                variance_type=sched_params["variance_type"],
                clip_sample=False,
                clip_sample_range=1.0,
                prediction_type=sched_params["prediction_type"],
                thresholding=False,
                dynamic_thresholding_ratio=0.995,
                sample_max_value=1.0
            )
        elif sched_params["name"] == "ddim":
            # From: https://huggingface.co/docs/diffusers/api/schedulers/ddim
            #       https://huggingface.co/docs/diffusers/api/pipelines/ddim
            self.sched = DDIMScheduler(
                num_train_timesteps=sched_params["num_train_timesteps"],
                beta_start=sched_params["beta_start"],  # 0.0001
                beta_end=sched_params["beta_end"],  # 0.02
                beta_schedule=sched_params["beta_schedule"],
                trained_betas=None,
                clip_sample=False,
                clip_sample_range=1.0,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type=sched_params["prediction_type"],
                thresholding=False,
                dynamic_thresholding_ratio=0.995,
                sample_max_value=1.0,
                rescale_betas_zero_snr=sched_params.get("rescale_betas_zero_snr", False),
                timestep_spacing=sched_params.get("timestep_spacing", "leading"),
            )
        # elif sched_params["name"] == "edm":
        #     if EDMEulerScheduler is None:
        #         diffusers_version = diffusers.__version__
        #         raise ImportError(
        #             "EDMEulerScheduler not present in diffusers library"
        #             f" version {diffusers_version}"
        #         )
        #     # From: https://huggingface.co/docs/diffusers/en/api/schedulers/edm_euler
        #     self.sched = EDMEulerScheduler(
        #         sigma_min=sched_params.get("sigma_min", 0.002),
        #         sigma_max=sched_params.get("sigma_max", 80.0),
        #         sigma_data=sched_params.get("sigma_data", 0.5),
        #         sigma_schedule='karras',
        #         num_train_timesteps=sched_params["num_train_timesteps"],
        #         prediction_type=sched_params["prediction_type"],
        #         rho=sched_params.get("rho", 7.0)
        #     )
        else:
            raise NotImplementedError(sched_params["name"])

        self.pred_type = sched_params["prediction_type"]
        self.sched_params = sched_params

        # Self-conditioning.
        self.use_sc = False if sc_params is None else sc_params["use"]
        if self.use_sc:
            self.sc_train_p = sc_params["train_p"]
        else:
            self.sc_train_p = None

        # xyz loss (by default it is not used).
        self.extra_loss = extra_loss
        if extra_loss and extra_loss["name"] in ("cg_e", ):
            self.use_xyz_loss = True
            self.xyz_loss_t_on = extra_loss["active_t"]
        else:
            self.use_xyz_loss = False
            self.xyz_loss_t_on = None

        self.decoder = None
        self.xl_enc_std_scaler = None

    def set_decoder(self, decoder, enc_std_scaler):
        self.decoder = decoder
        self.xl_enc_std_scaler = enc_std_scaler


    def sample_time(self, x0, batch_size):
        if self.sched_params["name"] in ("ddpm", "ddim"):
            if self.sched_params.get("max_train_timestep") is not None:
                max_timestep = self.sched_params["max_train_timestep"]
            else:
                max_timestep = self.sched_params["num_train_timesteps"]
            t = torch.randint(0, max_timestep,
                              (batch_size,),
                              device=x0.device, dtype=torch.long)
        # elif self.sched_params["name"] == "edm":
        #     t = torch.randint(0, self.sched_params["num_train_timesteps"],
        #                       (batch_size,),
        #                       device=x0.device, dtype=torch.long)
        #     # timesteps = self.sched.timesteps.to(x0.device)
        #     # t = timesteps[t]
        else:
            raise KeyError(self.sched_params["name"])
        return t


    def loss(self,
             batch,
             noise: Optional[torch.Tensor] = None,
             reduction: str = "mean"):
        """
        TODO.
        """

        # Get batch size
        batch_size = batch.num_graphs
        # Encodings at time zero.
        x0 = batch.z

        # Get random $t$ for each sample in the batch.
        t = self.sample_time(x0, batch_size)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is not None:
            raise NotImplementedError()
        noise = torch.randn_like(batch.z)

        # Sample $x_t$ for $q(x_t|x_0)$.
        xt = self.sched.add_noise(x0, noise, t)

        # Regular diffusion modeling.
        if not self.use_sc:
            # Get the model output.
            model_out = self.sched.scale_model_input(
                sample=self.eps_model(xt=xt, t=t, batch=batch),
                timestep=t
            )
        # Diffusion modeling with self-conditioning.
        else:
            raise NotImplementedError()

        if self.pred_type == "epsilon":
            target = noise
        elif self.pred_type == "sample":
            target = x0
        elif self.pred_type == "v_prediction":
            target = self.sched.get_velocity(
                sample=x0, noise=noise, timesteps=t
            )
        else:
            raise NotImplementedError()

        # Compute the main, MSE-based loss.
        if self.loss_type == 'l1':
            loss = F.l1_loss(target, model_out, reduction=reduction)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, model_out, reduction=reduction)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(target, model_out, reduction=reduction)
        else:
            raise NotImplementedError()

        # Compute the xyz structure loss.
        if self.use_xyz_loss:
            include_dec = True  # self.extra_loss.get("include_dec", False)
            is_active = t.min() <= self.xyz_loss_t_on
            if is_active or include_dec:
                # Compute the loss for each element of the batch.
                xyz_loss = self.compute_xyz_loss(xt, x0, t, model_out, batch)
                terms = {}
                terms["gm_loss"] = loss.item()
                # Average.
                if is_active:
                    mask_t = t <= self.xyz_loss_t_on
                    # Log.
                    terms["xyz_loss"] = xyz_loss[mask_t].mean().item()
                    # Only average over active elements.
                    mask_t = mask_t.float()
                    xyz_loss = torch.sum(xyz_loss*mask_t)/mask_t.sum()
                    extra_loss_w = self.extra_loss["weight"]
                else:
                    # Average over all elements, it will be zeroed-out in any
                    # case.
                    terms["xyz_loss"] = float('nan')
                    xyz_loss = torch.mean(xyz_loss)
                    extra_loss_w = 0.0
                loss += xyz_loss*extra_loss_w
            else:
                terms = {"gm_loss": loss.item()}
            return loss, terms
        else:
            return loss

    
    def compute_xyz_loss(self, xt, x0, t, model_out, batch):
        # Reconstruct the predicted encoding at time 0.
        x0_hat = self.reconstruct_pred_original_sample(
            sample=xt, t=t.view(-1, 1, 1), model_output=model_out
        )
        if self.xl_enc_std_scaler is not None:
            u_scaler = self.xl_enc_std_scaler["u"].to(xt.device)
            s_scaler = self.xl_enc_std_scaler["s"].to(xt.device)
            x0_hat = x0_hat*s_scaler + u_scaler
        # Decode it with the decoder.
        s_hat = self.decoder.nn_forward(x0_hat, batch)
        if self.extra_loss["name"] == "cg_e":
            # CG loss.
            if not has_openfold:
                raise ImportError("OpenFold is not installed")
            # Compute non-bonded Ca-Ca clash energy.
            w_vdw = self.extra_loss.get("vdw_weight", 1.0)
            e_vdw = compute_e_vdw(
                s_hat, batch, eps=1e-12, reduce=None
            )*w_vdw
            # Compute adjacent Ca-Ca bond energy.
            w_bond = self.extra_loss.get("bond_weight", 1.0)
            e_bond = compute_e_bond(
                s_hat, batch, eps=1e-9, reduce=None
            )*w_bond
            # Get the total energy.
            e_tot = e_vdw + e_bond

        else:
            raise KeyError(self.extra_loss["name"])
        
        # e_tot = e_tot[t <= self.xyz_loss_t_on]
        # if e_tot.shape[0] == 0:
        #     raise ValueError()
        # loss = e_tot.mean()
        
        return e_tot

    def reconstruct_pred_original_sample(self, sample, t, model_output):
        if self.sched_params["name"] == "ddpm":

            prev_t = self.sched.previous_timestep(t)

            # 1. compute alphas, betas
            alpha_prod_t = self.sched.alphas_cumprod[t]
            sched_one = self.sched.one.to(device=t.device)
            # alpha_prod_t_prev = self.sched.alphas_cumprod[prev_t] if prev_t >= 0 else self.sched.one
            alpha_prod_t_prev = torch.where(
                prev_t >= 0, self.sched.alphas_cumprod[prev_t], sched_one
            )
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.sched.config.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.sched.config.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.sched.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.sched.config.prediction_type} must be one of `epsilon`, `sample` or"
                    " `v_prediction`  for the DDPMScheduler."
                )
        else:
            raise KeyError(self.sched_params["name"])
        
        # if self.xl_enc_std_scaler is not None:
        #     device = pred_original_sample.device
        #     u_scaler = self.xl_enc_std_scaler["u"].to(device=device)
        #     s_scaler = self.xl_enc_std_scaler["s"].to(device=device)
        #     pred_original_sample = pred_original_sample*s_scaler + u_scaler

        return pred_original_sample


    # @torch.no_grad()
    def sample(self, batch, x_0=None, t_start=None,
               n_steps=None, variance=None,
               use_grad=False, use_cache=False,
               *args, **kwargs):
        
        if x_0 is not None or t_start is not None:
            raise NotImplementedError()

        self._setup_sampling_sched(n_steps, variance)

        model = self.get_sample_model()
        x_t = torch.randn_like(batch.z)
        batch_size = batch.num_graphs
        iteration = 0
        for i in self.sched.timesteps:
            t = torch.full((batch_size, ), i,
                           device=batch.z.device, dtype=torch.long)
            grad_context = torch.no_grad
            with grad_context():
                with self.sampling_context():
                    if not use_cache:
                        model_out = model(xt=x_t, t=t, batch=batch)
                    else:
                        if iteration == 0:
                            model_out, cache = model(xt=x_t, t=t, batch=batch, get_cache=True)
                        else:
                            model_out = model(xt=x_t, t=t, batch=batch, cache=cache)
                noisy_residual = self.sched.scale_model_input(
                    sample=model_out,
                    timestep=t
                )
            x_t = self.sched.step(noisy_residual, i, x_t).prev_sample
            iteration += 1
        out = x_t
        return out

    @contextmanager
    def sampling_context(self):
        ctx = nullcontext if self.ema is None else self.ema.average_parameters
        yield ctx

    def _setup_sampling_sched(self, n_steps, variance):
        if self.sched_params["name"] == "ddpm":
            if n_steps is not None:
                self.sched.set_timesteps(n_steps)
            if variance is not None:
                # Options: ("fixed_small", "fixed_small_log",
                #           "fixed_large", "fixed_large_log")
                self.sched.variance_type = variance
        elif self.sched_params["name"] == "ddim":
            self.sched.set_timesteps(n_steps if n_steps is not None \
                                     else self.sched.config.num_train_timesteps)
        else:
            raise NotImplementedError()

        if not self.pred_type in ("epsilon", "sample", "v_prediction"):
            raise NotImplementedError()


    # EXTERNAL_POTENTIAL +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    bias_stages = ("guidance", "correct", "relax")

    def _get_bias_stage_config(self, config, stage):

        if not stage in self.bias_stages:
            raise KeyError(stage)

        # Common.
        out = {}

        func = get_potential(config["func"])
        if isinstance(func, types.FunctionType):
            pass
        elif isinstance(func, type):
            func = func(**config["args"])
        else:
            raise TypeError(type(func))
        out["func"] = func

        out["w_bias_base"] = config.get("w_bias_base", 1.0)
        out["func_params"] = config.get("func_params", {})
        out["rescale_grad"] = config.get("rescale_grad", True)
        out["clip"] = config.get("clip", None)
        out["score_clashes"] = config.get("score_clashes", False)
        # weight_mode = config.get("weight_mode", "linear")

        # Stage-specific.
        if stage == "guidance":
            out["stop_grad_at_eps"] = config.get("stop_grad_at_eps", False)

        if stage in ("guidance", "correct"):
            out["active_t"] = config.get("active_t", 350) # int(os.getenv("SAM_BIAS_ACTIVE_T", "350"))  # 250, 500
        
        if stage in ("correct", "relax"):
            out["steps"] = config["steps"]
            out["step_size"] = config["step_size"]
            out["w_init"] = config.get("w_init")
            out["opt"] = config.get("opt", "sgd")
        
        return out


    def sample_bias(self, batch, x_0=None, t_start=None,
                    n_steps=None, variance=None,
                    get_traj=False,
                    bias: dict = {},
                    use_cache: bool = False,
                    *args, **kwargs):
        
        # Setup input.
        if not any([stage in bias for stage in self.bias_stages]):
            if not get_traj:
                raise ValueError(
                    "The bias argument should contain at least one the"
                    " following keys: {}".format(self.bias_stages)
                )

        if x_0 is not None or t_start is not None:
            raise NotImplementedError()
        
        self._setup_sampling_sched(n_steps, variance)

        if "general" in bias:
            general = bias["general"]
        else:
            general = {"verbose": False}

        if "guidance" in bias:
            guidance = self._get_bias_stage_config(bias["guidance"], "guidance")
        else:
            guidance = {}
        
        if "correct" in bias:
            correct = self._get_bias_stage_config(bias["correct"], "correct")
        else:
            correct = {}
        
        if "relax" in bias:
            relax = self._get_bias_stage_config(bias["relax"], "relax")
        else:
            relax = {}
        
        # Configure the diffusion process.
        model = self.get_sample_model()
        x_t = torch.randn_like(batch.z)
        batch_size = batch.num_graphs
        
        traj = [x_t.unsqueeze(0)]

        tot_steps = self.sched.config.num_train_timesteps

        timesteps = self.sched.timesteps
        # timesteps = torch.cat(
        #     [torch.tensor(tot_steps-1, dtype=torch.long)[None], timesteps]
        # )
        # timesteps = torch.cat([timesteps, torch.zeros(20).long()])
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(batch.z.device)
        
        # Run reverse diffusion.
        iteration = 0
        for i in timesteps:
            # if i < 40:
            #     print("- Truncate!")
            #     break
            t = torch.full((batch_size, ), i,
                           device=batch.z.device, dtype=torch.long)
            
            if general["verbose"]:
                print(f"\n# timestep={i}")

            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if guidance and i <= guidance["active_t"]:

                _x_t = x_t.clone()
                _x_t.requires_grad = True
                with self.sampling_context():
                    if not use_cache:
                        model_out = model(xt=_x_t, t=t, batch=batch)
                    else:
                        if iteration == 0:
                            model_out, cache = model(xt=_x_t, t=t, batch=batch, get_cache=True)
                        else:
                            model_out = model(xt=_x_t, t=t, batch=batch, cache=cache)
                    if guidance["stop_grad_at_eps"]:
                        model_out = model_out.detach()
                _x_0 = self.reconstruct_pred_original_sample(_x_t, t.view(-1, 1, 1), model_out)
                if self.enc_std_scaler is not None:
                    _x_0 = _x_0*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]
                xyz_0 = self.decoder.nn_forward(_x_0, batch)
                
                #===
                self._report_clashes(xyz_0, batch, guidance["score_clashes"])
                #===

                alpha_prod_t = self.sched.alphas_cumprod[i]
                beta_prod_t = 1 - alpha_prod_t
                sqrt_beta_prod_t = beta_prod_t ** (0.5)
                w_bias = guidance["w_bias_base"]*sqrt_beta_prod_t
                
                #---
                energy_t = guidance["func"](
                    xyz_0, batch, **guidance["func_params"]
                )
                #---

                energy_t.backward()
                if guidance["rescale_grad"] and self.enc_std_scaler is not None:
                    scaled_grad = _x_t.grad / self.enc_std_scaler["s"]
                else:
                    scaled_grad = _x_t.grad
                if guidance["clip"] is not None:
                    scaled_grad = torch.clamp(
                        scaled_grad, min=-guidance["clip"], max=guidance["clip"]
                    )
                bias = scaled_grad*w_bias
                use_bias = True

            else:
                bias = 0
                use_bias = False

                with torch.no_grad():
                    with self.sampling_context():
                        if not use_cache:
                            model_out = model(xt=x_t, t=t, batch=batch)
                        else:
                            if iteration == 0:
                                model_out, cache = model(xt=x_t, t=t, batch=batch, get_cache=True)
                            else:
                                model_out = model(xt=x_t, t=t, batch=batch, cache=cache)

            # EXTERNAL_POTENTIAL ###########################################
            if general["verbose"]:
                print("- eps:", model_out.abs().mean().item())
                if use_bias:
                    print("- bias:", bias.abs().mean().item())

            noisy_residual = self.sched.scale_model_input(
                sample=model_out + bias,
                timestep=t
            )
            ################################################################
            x_t = self.sched.step(noisy_residual, i, x_t).prev_sample
            x_t = x_t.detach()
            
            # EXTERNAL_POTENTIAL ###############################################
            if correct and i <= correct["active_t"]:
                x_t = self.relax_steps(
                    x_t=x_t,
                    batch=batch,
                    func=correct["func"],
                    func_params=correct["func_params"],
                    w_bias_base=correct["w_bias_base"],
                    steps=correct["steps"],
                    step_size=correct["step_size"],
                    rescale_grad=correct["rescale_grad"],
                    w_init=correct["w_init"],
                    opt=correct["opt"],
                    score_clashes=correct["score_clashes"],
                    verbose=general["verbose"],
                    stage="correct"
                )

            ####################################################################
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if get_traj:
                traj.append(x_t.unsqueeze(0))
            ####################################################################
            iteration += 1

        # EXTERNAL_POTENTIAL ###################################################
        if relax:
            x_t = self.relax_steps(
                x_t=x_t,
                batch=batch,
                func=relax["func"],
                func_params=relax["func_params"],
                w_bias_base=relax["w_bias_base"],
                steps=relax["steps"],
                step_size=relax["step_size"],
                rescale_grad=relax["rescale_grad"],
                w_init=relax["w_init"],
                opt=relax["opt"],
                score_clashes=relax["score_clashes"],
                verbose=general["verbose"],
                stage="relax"
            )
        ########################################################################
        
        out = x_t
        # EXTERNAL_POTENTIAL ###################################################
        if not get_traj:
            return out
        else:
            return {"enc": out, "traj": torch.cat(traj, axis=0)}
        ########################################################################


    def _report_clashes(self, xyz, batch, report):
        if report:  ###
            clashes = compute_clashes(xyz, batch)  ###
            print("+ Ca-Ca clashes:", clashes.item())  ###


    def relax_steps(self,
            x_t: torch.Tensor,
            batch: torch.Tensor,
            func: Callable,
            func_params: dict,
            w_bias_base: float,
            steps: int,
            step_size: int,
            rescale_grad: bool = True,
            w_init: float = None,
            opt: str = "sgd",
            score_clashes: bool = False,
            verbose: bool = False,
            stage: str = "minimize"
        ):

        # if w_init is not None:
        #     _x_init = x_t.clone()
        #     if self.enc_std_scaler is not None:
        #         _x_init = _x_init*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]

        #++++++++++++++++++++++++++++++
        # import time
        # import mdtraj
        # from sam.data.aa_topology import get_traj_list
        # from sam.evaluation.stabilitas import mscore_stabilititas
        # import pickle
        #++++++++++++++++++++++++++++++

        _x_t = x_t.clone()
        if self.enc_std_scaler is not None:
            _x_t = _x_t*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]
        _x_t.requires_grad = True

        if opt == "sgd":
            optimizer = torch.optim.SGD([_x_t], lr=step_size, momentum=0.9)
        elif opt == "gd":
            optimizer = torch.optim.SGD([_x_t], lr=step_size, momentum=0.0)
        elif opt == "adam":
            optimizer = torch.optim.Adam([_x_t], lr=step_size)
        elif opt == "lbfgs":
            optimizer = torch.optim.LBFGS([_x_t], lr=step_size, max_iter=3)
        else:
            raise KeyError(opt)

        #++++++++++++++++++++++++++++++
        # traj_gen = []
        # xyz_gen = []
        # ofo_gen = []
        #++++++++++++++++++++++++++++++

        with torch.no_grad():
            xyz_init = self.decoder.nn_forward(_x_t, batch)
            if verbose:
                ca_pos_init = xyz_init["positions"][-1,:,:,1,:]
                ca_dist_init = torch.cdist(ca_pos_init, ca_pos_init)
        
        def closure():
            optimizer.zero_grad()
            if verbose:
                print(f"# {stage} step", i)  ###
            xyz_t = self.decoder.nn_forward(_x_t, batch)
            #@@@@@@@@@@@@@@@@
            if i != 0:
                if verbose:
                    ca_pos = xyz_t["positions"][-1,:,:,1,:]
                    ca_dist = torch.cdist(ca_pos, ca_pos)
                    n_res = ca_pos.shape[1]
                    ti = torch.triu_indices(n_res, n_res, offset=1)
                    drmsd = torch.square(
                        ca_dist_init[:,ti[0],ti[1]] - ca_dist[:,ti[0],ti[1]]
                    ).mean(axis=-1).sqrt()
                    print("- dRMSD", drmsd.mean().item())
            #@@@@@@@@@@@@@@@@
            #++++++++++++++++++++++++++
            # xyz_gen.append(xyz_t)
            # print("- Current traj:", len(traj_gen))
            #++++++++++++++++++++++++++
            #===
            self._report_clashes(xyz_t, batch, score_clashes)
            #===
            energy_t = func(xyz_t, batch, **func_params)
            """
            if i != 0:
                energy_restraints = drmsd.sum(dim=0)*1000.0
                energy_t = energy_t + energy_restraints
            """
            # if w_init is not None:
            #     rst_init = torch.square(_x_init - _x_t).sum(dim=[1, 2]).sum(dim=0)
            #     energy_t += rst_init*w_init
            w_bias = w_bias_base * 1.0
            # energy_t.grad = energy_t*w_bias  # ?????????????????????????????????
            energy_t = energy_t*w_bias
            energy_t.backward()
            # if rescale_grad and self.enc_std_scaler is not None:
            #     _x_t.grad = _x_t.grad / self.enc_std_scaler["s"]
            # else:
            #     _x_t.grad = _x_t.grad
            return energy_t

        for i in range(steps):
            optimizer.step(closure)
        
        #++++++++++++++++++++++++++++++
        # sta_args = {"func": "l2", "sel": "aa"}
        # for sm_i in xyz_gen:
        #     energy_i = func(sm_i, batch, **func_params)
        #     sm_i["positions"] = sm_i["positions"][-1][-2:-1]
        #     ofo_gen.append({
        #         "positions": sm_i["positions"].cpu(), 
        #         # "atom14_atom_exists": sm_i["atom14_atom_exists"].cpu(),
        #         # "atom14_elements": sm_i["atom14_elements"].cpu()
        #     })
        #     traj_i = get_traj_list(sm_i)
        #     sta_i = np.mean(mscore_stabilititas(mdtraj.join(traj_i), **sta_args))
        #     print("mscore:", sta_i)
        #     traj_gen.extend(traj_i)
        # traj_gen = mdtraj.join(traj_gen)
        # time_ = time.time()
        # traj_gen.save(f"/home/giacomo/projects/sam/git/idpsam/{time_}.pdb")
        # with open(f"/home/giacomo/projects/sam/git/idpsam/{time_}.pkl", "wb") as o_fh:
        #     pickle.dump(ofo_gen, o_fh)
        #++++++++++++++++++++++++++++++

        _x_t = _x_t.detach()
        if self.enc_std_scaler is not None:
            _x_t = (_x_t-self.enc_std_scaler["u"])/self.enc_std_scaler["s"]

        return _x_t


def compute_clashes(xyz, batch, threshold=0.39, reduce="mean"):
    if isinstance(xyz, dict):
        # Get Ca atoms positions.
        pos = xyz["positions"][-1,:,:,:,:]
        ca_pos = pos[:,:,1,:]*0.1
    else:
        raise NotImplementedError("TODO")
    # Count the clashes.
    num_beads = ca_pos.shape[1]
    triu_ids = torch.triu_indices(num_beads, num_beads, offset=3)
    ca_dist = torch.cdist(ca_pos, ca_pos, p=2.0)
    ca_dist = ca_dist[:,triu_ids[0],triu_ids[1]]
    clashes = ca_dist < threshold
    clashes = clashes.float()
    clashes = clashes.sum(dim=1)
    if reduce == "mean":
        return clashes.mean(dim=0)
    else:
        raise NotImplementedError("TODO")


########################
# Old biasing version. #
########################

# # EXTERNAL_POTENTIAL ###############################################
# if i <= active_t and biasing:  # 500

#     _x_t = x_t.clone()
#     if self.enc_std_scaler is not None:
#         _x_t = _x_t*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]
#     _x_t.requires_grad = True
#     xyz_t = self.decoder.nn_forward(_x_t, batch)
#     if score_vdw:  ###
#         e_vdw = compute_e_vdw(xyz_t, batch)  ###
#         print("+ e_vdw:", e_vdw.item())  ###
#     #---
#     energy_t = func(xyz_t, batch, **func_params)
#     #---
#     energy_t.backward()
#     if weight_mode == "linear":
#         w_bias = w_bias_base * (tot_steps - i)/tot_steps  # 250.0
#     elif weight_mode == "const":
#         w_bias = w_bias_base
#     elif weight_mode == "alpha":
#         alpha_prod_t = self.sched.alphas_cumprod[i]
#         beta_prod_t = 1 - alpha_prod_t
#         sqrt_beta_prod_t = beta_prod_t ** (0.5)
#         w_bias = w_bias_base*sqrt_beta_prod_t
#     else:
#         raise KeyError(weight_mode)
#     if rescale_grad and self.enc_std_scaler is not None:
#         scaled_grad = _x_t.grad / self.enc_std_scaler["s"]
#     else:
#         scaled_grad = _x_t.grad
#     if clip is not None:
#         scaled_grad = torch.clamp(scaled_grad, min=-clip, max=clip)
#     bias = scaled_grad*w_bias
#     use_bias = True

#     if balancing_params:
#         ### if balancing_params["mode"] == "linear":
#         ###     pi_bias = (active_t - i) / active_t
#         ###     pi_eps = 1.0 - pi_bias
#         ### elif balancing_params["mode"] == "const":
#         ###     pi_bias = 1.0
#         ###     pi_eps = 0.0
#         ### else:
#         ###     raise KeyError(balancing_params["mode"])
#         pass
#     else:
#         pi_eps = 1.0
#         pi_bias = 1.0

# else:
#     bias = 0
#     use_bias = False
#     pi_eps = 1.0
#     pi_bias = 1.0
# ####################################################################

# with torch.no_grad():
#     # EXTERNAL_POTENTIAL ###########################################
#     with self.sampling_context():
#         model_out = model(xt=x_t, t=t, batch=batch)
#     print("- eps:", model_out.abs().mean().item(), pi_eps)  ###
#     if use_bias:  ###
#         print("- bias:", bias.abs().mean().item(), pi_bias)  ###
#     noisy_residual = self.sched.scale_model_input(
#         sample=model_out + bias,  # sample=model_out*pi_eps + bias*pi_bias,
#         timestep=t
#     )
#     ################################################################
# x_t = self.sched.step(noisy_residual, i, x_t).prev_sample

# # EXTERNAL_POTENTIAL ###############################################
# if i <= active_t and correct_params:
#     x_t = self.relax_steps(
#         x_t=x_t,
#         batch=batch,
#         func=func,
#         func_params=func_params,
#         w_bias_base=w_bias_base,
#         steps=correct_params["steps"],
#         step_size=correct_params["step_size"],
#         rescale_grad=rescale_grad,
#         w_init=correct_params.get("w_init"),
#         score_vdw=score_vdw
#     )


######################
# Old relax version. #
######################

# if w_init is not None:
#     _x_init = x_t.clone()
#     if self.enc_std_scaler is not None:
#         _x_init = _x_init*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]

# for i in range(steps):
#     print("- Relax step", i)  ###
#     _x_t = x_t.clone()
#     if self.enc_std_scaler is not None:
#         _x_t = _x_t*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]
#     _x_t.requires_grad = True
#     xyz_t = self.decoder.nn_forward(_x_t, batch)
#     if True:  # if i == 0 and score_vdw:  ###
#         e_vdw = compute_e_vdw(xyz_t, batch)  ###
#         print("+ e_vdw:", e_vdw.item())  ###
#     energy_t = func(xyz_t, batch, **func_params)
#     if w_init is not None:
#         rst_init = torch.square(_x_init - _x_t).sum(dim=[1, 2]).sum(dim=0)
#         energy_t += rst_init*w_init
#     energy_t.backward()
#     if rescale_grad and self.enc_std_scaler is not None:
#         scaled_grad = _x_t.grad / self.enc_std_scaler["s"]
#     else:
#         scaled_grad = _x_t.grad
#     w_bias = w_bias_base * 1.0
#     bias = scaled_grad*w_bias
#     print("- bias:", bias.abs().mean().item())  ###
#     x_t = x_t - bias*step_size
# return x_t