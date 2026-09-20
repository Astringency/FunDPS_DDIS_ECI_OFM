"""Interface adapters; all network, integration and guidance code stays upstream."""
from contextlib import contextmanager
import copy
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

from evaluate_flow import observation_indices
from train_flow import build_prior


class FlowNoise:
    def __init__(self, device, gp=None):
        self.device = device
        self.gp = gp

    def sample(self, grid, dims, n_samples=1):
        if self.gp is None:
            return torch.randn(n_samples, *dims, device=self.device, dtype=torch.float32)
        return self.gp.sample(list(dims[1:]), n_samples=n_samples,
                              n_channels=dims[0]).to(self.device, torch.float32)


class FM4PDEVelocityAdapter(torch.nn.Module):
    """Expose OFM's (time, state) interface to the native FM4PDE runner."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, extra=None):
        if extra:
            raise ValueError('This OFM prior has no scalar or category conditioning')
        return self.model(t, x)


class ECIVelocityAdapter:
    """Expose native FM4PDE conditioning and (state, time) calls to ECI."""
    def __init__(self, net, extra):
        self.net, self.extra = net, extra

    def __call__(self, t, x):
        from sampling.sampler_wrappers import _call_velocity_model
        return _call_velocity_model(self.net, x, t, self.extra)


def load_eci(repo):
    # Both projects use a top-level package named models. Keep ECI isolated.
    name = 'shared_prior_official_eci'
    package = types.ModuleType(name)
    package.__path__ = [str(Path(repo) / 'models')]
    sys.modules[name] = package
    functional = importlib.import_module(name + '.functional')
    constraints = importlib.import_module(name + '.constraints')
    return functional.FFM, constraints.DirichletCondition


def load_prior(args, channels):
    # Import the native FM4PDE namespace before OFM adds its own source path.
    sys.path.insert(0, str(args.fm4pde))
    from data.transform import PDEStandardizer
    from sampling.model_io import load_fm4pde_checkpoint_bundle
    import sampling.runner

    if args.prior == 'fm4pde':
        net, normalizer, payload = load_fm4pde_checkpoint_bundle(
            str(args.checkpoint), args.pde, args.device, wrap=True, model_profile='auto')
        return net, normalizer, payload, FlowNoise(args.device), None
    model, operator_prior = build_prior(args.ofm, 128, args.device, channels=channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    manifest = json.loads((args.source / 'manifest.json').read_text())
    normalizer = PDEStandardizer(torch.tensor(manifest['stats']['mean']),
        torch.tensor(manifest['stats']['std']) / 0.5, pde=args.pde)
    payload = {'num_channels': channels, 'normalizer': normalizer.state_dict(),
               'normalization': {'type': 'OFM training coordinates', 'scale': 0.5},
               'selected_inference_weight': 'validation_selected_raw',
               'selected_architecture_family': 'official_OFM_FNO',
               'model_config': {'in_channels': channels, 'num_classes': None},
               'data_shape': [channels, 128, 128]}
    return FM4PDEVelocityAdapter(model), normalizer, payload, FlowNoise(args.device, operator_prior.gp), operator_prior


def case_ground_truth(saved, index, device):
    from sampling.data import PDEGroundTruth

    def select(value):
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == 100:
                value = value[index:index + 1]
            return value.to(device)
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        return copy.deepcopy(value)

    values = select(saved)
    values['metadata'].update(offset=index, batch_size=1, sample_offsets=[index], sample_ids=[str(index)])
    temporal = values['metadata'].get('full_trajectory_fd')
    if temporal and temporal.get('shape'):
        temporal['shape'][0] = 1
    return PDEGroundTruth(**values)


def common_masks(truth, index, seed=0):
    from sampling.masks import PairMasks
    indices = observation_indices(seed)[index]
    mask = torch.zeros_like(truth.sol)
    mask.reshape(-1)[torch.as_tensor(indices, device=mask.device)] = 1
    assert int(mask.sum()) == 500
    return PairMasks(torch.zeros_like(truth.coef), mask,
        {'num_obs': 500, 'active_observations': 500, 'active_channel': 'solution',
         'sensor_mode': 'common_ddis_random', 'mask_seed': seed, 'sample_id': index,
         'shared_across_methods_and_priors': True}), indices


def native_model_extra(payload, truth, config):
    from sampling.runner import _class_conditioning_for_sampling, _scalar_conditioning_for_sampling
    scalar, _ = _scalar_conditioning_for_sampling(checkpoint_payload=payload, gt=truth,
                                                  config=config, device=config.device)
    classes, _ = _class_conditioning_for_sampling(checkpoint_payload=payload, pde=config.pde,
        batch_size=1, device=config.device, cfg_scale=config.cfg_scale)
    return {**classes, **(scalar or {})} or None


@contextmanager
def native_noise_provider(noise, enabled):
    """Bind the trained GP to both native noise draws, restoring them afterwards.

    FM4PDE's step, extrapolation, guidance and time grid remain unchanged.
    Ordinary FM uses the untouched native iid Gaussian implementation.
    """
    if not enabled:
        yield
        return
    import sampling.runner as runner
    import sampling.sampler_wrappers as wrappers
    original_initial = runner._sample_initial_noise
    original_bridge = wrappers._stochastic_bridge_noise_like

    def draw(template, source_batch_size=None, source_indices=None):
        batch = int(template.shape[0])
        count = batch if source_batch_size is None else int(source_batch_size)
        indices = list(range(batch)) if source_indices is None else list(source_indices)
        if count < batch or len(indices) != batch or any(i < 0 or i >= count for i in indices):
            raise ValueError('Invalid source batch selection for OFM GP noise')
        sample = noise.sample(None, list(template.shape[1:]), n_samples=count).to(template)
        return sample[indices]

    def initial(config, ground_truth, device):
        return draw(ground_truth.pair, config.initial_noise_source_batch_size,
                    config.initial_noise_source_indices or None)

    def bridge(x_cur, *, device, source_batch_size=None, source_indices=None):
        return draw(x_cur, source_batch_size, source_indices)

    runner._sample_initial_noise = initial
    wrappers._stochastic_bridge_noise_like = bridge
    try:
        yield
    finally:
        runner._sample_initial_noise = original_initial
        wrappers._stochastic_bridge_noise_like = original_bridge
