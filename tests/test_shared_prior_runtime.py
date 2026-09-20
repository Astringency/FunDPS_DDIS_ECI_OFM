"""Integration checks for adapters against the unchanged official samplers."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
FM = Path(os.environ['FM4PDE_SOURCE'])
ECI = Path(os.environ['ECI_SOURCE'])
sys.path[:0] = [str(ROOT / 'adapters'), str(FM)]

from data.transform import PDEStandardizer
import sampling.runner as runner
import sampling.sampler_wrappers as wrappers
from shared_prior_runtime import (ECIVelocityAdapter, FM4PDEVelocityAdapter,
    FlowNoise, common_masks, load_eci, native_noise_provider)


class LinearVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, t, x):
        return self.weight * x + t


class GPSpy:
    def __init__(self):
        self.calls = []

    def sample(self, dims, n_samples, n_channels):
        self.calls.append((dims, n_samples, n_channels))
        return torch.full((n_samples, n_channels, *dims), 0.375)


class OfficialInterfaceTests(unittest.TestCase):
    def test_input_gradient_and_parameter_freezing(self):
        model = LinearVelocity().eval().requires_grad_(False)
        net = FM4PDEVelocityAdapter(model)
        x = torch.randn(1, 2, 8, 8, requires_grad=True)
        y = ECIVelocityAdapter(net, None)(torch.tensor(0.3), x)
        gradient, = torch.autograd.grad(y.sum(), x)
        torch.testing.assert_close(gradient, torch.full_like(x, 0.25))
        self.assertIsNone(model.weight.grad)
        self.assertFalse(model.weight.requires_grad)

    def test_both_native_noise_draws_and_restoration(self):
        old_initial, old_bridge = runner._sample_initial_noise, wrappers._stochastic_bridge_noise_like
        template = torch.zeros(1, 2, 8, 8)
        gp = GPSpy()
        noise = FlowNoise('cpu', gp)
        config = SimpleNamespace(initial_noise_source_batch_size=None, initial_noise_source_indices=[])
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with native_noise_provider(noise, True):
                initial = runner._sample_initial_noise(config, SimpleNamespace(pair=template), 'cpu')
                endpoint, proposal = wrappers._stochastic_step(FM4PDEVelocityAdapter(LinearVelocity()),
                    template, torch.tensor(0.), torch.tensor(0.25), torch.tensor(0.25), 'euler', 'cpu', None)
                torch.testing.assert_close(initial, torch.full_like(template, 0.375))
                torch.testing.assert_close(proposal, 0.75 * initial + 0.25 * endpoint)
                self.assertEqual(len(gp.calls), 2)
                raise RuntimeError('intentional')
        self.assertIs(runner._sample_initial_noise, old_initial)
        self.assertIs(wrappers._stochastic_bridge_noise_like, old_bridge)

    def test_official_eci_masks_and_namespace_for_both_channel_counts(self):
        import models
        native_models = models
        FFM, Constraint = load_eci(ECI)
        self.assertIs(sys.modules['models'], native_models)
        for channels in (1, 2):
            target = torch.randn(1, channels, 128, 128)
            truth = SimpleNamespace(coef=target[:, :1], sol=target[:, -1:])
            masks, indices = common_masks(truth, 17)
            self.assertEqual(len(set(indices)), 500)
            self.assertEqual(int(masks.coef.sum()), 0)
            self.assertEqual(int(masks.sol.sum()), 500)
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[:, -1:] = masks.sol.bool()
            model = FM4PDEVelocityAdapter(LinearVelocity().eval().requires_grad_(False))
            instance = SimpleNamespace(model=ECIVelocityAdapter(model, None), gp=FlowNoise('cpu'))
            output = FFM.eci_sample(instance, 1, 4, 2, None, [channels, 128, 128], 'cpu',
                                    Constraint(value=target, mask=mask))
            torch.testing.assert_close(output[mask], target[mask])
            self.assertTrue(torch.isfinite(output).all())

    def test_ofm_coordinates_use_half_standardization(self):
        mean, std = torch.tensor([8., 0.005]), torch.tensor([4., 0.003])
        normalizer = PDEStandardizer(mean, std / 0.5)
        physical = torch.randn(2, 2, 8, 8)
        standardized = (physical - mean[None, :, None, None]) / std[None, :, None, None] * 0.5
        torch.testing.assert_close(normalizer.transform(physical), standardized)
        torch.testing.assert_close(normalizer.inverse_transform(standardized), physical)


if __name__ == '__main__':
    unittest.main()
