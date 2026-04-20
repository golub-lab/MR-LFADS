import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from mrlfads.utils.common_utils import flatten

def get_recon(model, area_name):
    sessions = sorted(model.current_batch.keys())
    area = model.areas[area_name]
    hps = model.hparams
    recon = []
    for s in sessions:
        recon.append( 
            area.output_dist(
                model.current_batch[s].encod_data[area_name][:,hps.ic_enc_seq_len:],
                model.outputs[area_name][s]
            ).sum(dim=(1, 2)).mean()
        )
    return recon

def get_holdout(model, area_name):
    sessions = sorted(model.current_batch.keys())
    area = model.areas[area_name]
    hn = []
    for s in sessions:
        hn.append( 
            model.holdout.compute_holdout(
                area_name,
                model.preds[area_name][s],
                area.output_dist,
                s
            ).sum(dim=(1, 2)).mean()
        )
    return hn

def get_pseudo_rsquared(model, sess=0):
    
    ic_enc_seq_len = model.hparams.ic_enc_seq_len
    r2_holdin_dict = {}, r2_holdout_dict = {}
    
    for area_name in model.area_names:
        
        data = model.raw_batch[sess].encod_data[area_name][:, ic_enc_seq_len:]
        hn_idx = model.hparams.hn_indices[area_name][sess]
        
        r2_in = model.areas[area_name].output_dist.pseudo_r2(data, model.outputs[area_name][sess], reduction=reduction)
        r2_out = model.areas[area_name].output_dist.pseudo_r2(data[..., hn_idx], model.preds[area_name][sess], reduction=reduction)
        r2_holdin_dict[area_name] = r2_in
        r2_holdout_dict[area_name] = r2_out
        
    return r2_holdin_dict, r2_holdout_dict

def fit_metrics(model, metrics):
    """Aggregate reconstruction and fit metrics into a summary dictionary.

    Extracts total/reconstruction/holdout loss, normalized KL terms,
    and R² across areas for held-in and held-out neurons.

    Args:
        model: `MRLFADS` class.
        metrics: Training/validation metrics dictionary.

    Returns:
        res: Dict containing scalar loss components and mean R² metrics.
    """

    hps = model.hparams
    r2_in, r2_out = get_pseudo_rsquared(model) # calculates r-squared values
    res = {
        'total': (metrics['valid/recon'] + metrics['valid/kl/u'] * hps.kl_co_scale + metrics['valid/kl/m'] * hps.kl_com_scale).item(),
        'recon': metrics['valid/recon'].item(),
        'hn': metrics['valid/hn'].item(),
        'kl(u)': metrics['valid/kl/u'].item() / hps.kl_co_scale,
        'kl(m)': metrics['valid/kl/m'].item() / hps.kl_com_scale,
        'r2(holdin)': np.mean(list(r2_in.values())),
        'r2(holdout)': np.mean(list(r2_out.values())),
    }
    return res

def cosine_similarity(model, true_m, true_u):
    """Compute cosine similarity between inferred and ground-truth connectivity.

    Compares model-inferred communication and inferred-input effectomes against
    ground-truth counterparts using cosine similarity. Diagonal (self-to-self)
    communication terms are excluded.

    Args:
        model: `MRLFADS` class.
        true_m: Ground-truth communication connectivity matrix.
        true_u: Ground-truth inferred-input effectome.

    Returns:
        sim_m: Cosine similarity between inferred and ground-truth communication
            connectivity (excluding diagonal terms).
        sim_u: Cosine similarity between inferred and ground-truth inferred-input
            effectomes.
    """

    pred_m, pred_u = volume(model, reduction=[-1, -2, -3])
    pred_m_flat, true_flat = [], []
    
    # Exclude diagonal terms
    for i in range(len(pred_u)):
        for j in range(len(pred_u)):
            if i != j:
                pred_m_flat.append(pred_m[i, j])
                true_flat.append(true_m[i, j])
                
    return 1-cosine(pred_m_flat, true_flat), 1-cosine(pred_u, true_u)

def volume(model, reduction=None):
    '''Extract communication and inferred-input volumes from a model over batch and time.
    
    Returns per-area communication volumes and inferred-input volumes. If `reduction`
    is provided, computes L2 norms over the specified axes instead of returning the
    raw volumes.
    
    Args:
        model: `MRLFADS` class.
        reduction: Optional iterable of axis indices to reduce over when computing
            L2 norms. If provided, the function returns L2 norms over those axes
            for both volumes. If None, returns raw volumes. Defaults to None.
            
    Returns:
        comm: Communication volume with shape
            (num_target_areas, num_source_areas, batch, time, com_dim), or the
            reduced L2 norm array if `reduction` is provided.
        inputs: Inferred-input volume with shape
            (num_areas, batch, time, co_dim), or the reduced L2 norm array if
            `reduction` is provided.
    '''
    hps = model.hparams
    num_areas = hps.num_other_areas + 1
    batch, time = model.save_var[model.area_names[0]].inputs.shape[:2]
    
    # Note: this function only works when all communication channels have equal dimensions
    # assert len(list(set([model.areas[model.area_names[i]].hparams.com_dim for i in range(num_areas)]))) == 1
    # co_dim = model.areas[model.area_names[0]].hparams.co_dim
    # com_dim = model.areas[model.area_names[0]].hparams.com_dim
    # volume, uvolume = np.zeros((num_areas, num_areas, batch, time, com_dim)), np.zeros((num_areas, batch, time, co_dim))
    
    # Locate and append communication, inferred inputs
    volume = [
        np.zeros((num_areas, batch, time, model.areas[model.area_names[i]].hparams.com_dim))
        for i in range(num_areas)
    ]
    uvolume = [
        np.zeros((batch, time, model.areas[model.area_names[i]].hparams.co_dim))
        for i in range(num_areas)
    ]
    
    for ia, (area_name, area) in enumerate(model.areas.items()):
        ahps = area.hparams
        num_other_areas_name = list(model.areas.keys())
        num_other_areas_name.pop(ia)

        inputs = model.save_var[area_name].inputs.detach().cpu()
        ci_size, com_dim, co_dim = ahps.ci_size, ahps.com_dim, ahps.co_dim
        _, com, co = torch.split(inputs, [ci_size, com_dim * hps.num_other_areas, co_dim], dim=2)

        for ioa in range(hps.num_other_areas):
            idx = ioa + 1 if ioa >= ia else ioa
            volume[ia][idx] = com[..., com_dim * ioa: com_dim * (ioa + 1)].numpy()
            
        uvolume[ia] = co.numpy()

    # Reduce dimensions of volumes
    if reduction is not None: 
        axes = tuple(reduction)
        com_reduction = [
            np.sqrt(np.sum(np.square(volume[ia]), axis=axes)) for ia in range(num_areas)
        ]
        co_reduction = [
            np.sqrt(np.sum(np.square(uvolume[ia]), axis=axes)) for ia in range(num_areas)
        ]
        return np.array(com_reduction), np.array(co_reduction)
    else: return volume, uvolume

class PolyRegression:
    def __init__(self, degree, alpha=0.0, tpe="ridge"):
        self.degree = degree
        self.alpha = alpha
        self.poly_features = PolynomialFeatures(degree=degree)
        
        if tpe == "ridge":
            self.reg = Ridge(alpha=alpha)
        elif tpe == "lasso":
            self.reg = Lasso(alpha=alpha)
        else:
            raise ValueError()

    def fit(self, X, y):
        X_poly = self.poly_features.fit_transform(X)
        self.reg.fit(X_poly, y)
        
    def ffit(self, X, y):
        self.fit(flatten(X), flatten(y))

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return self.reg.predict(X_poly)
    
    def fpredict(self, X):
        X_poly = self.poly_features.transform(flatten(X))
        return self.reg.predict(X_poly).reshape(*X.shape[:2], -1)

    def score(self, X, y): # X: prediction, y: true
        X_poly = self.poly_features.transform(X)
        return self.reg.score(X_poly, y)
    
    def fscore(self, X, y):
        pred = self.fpredict(X)
        return r2_score(flatten(y), flatten(pred))
    