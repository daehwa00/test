# Third-Party Assets and Licenses

This supplemental code uses and/or patches the following external open-source assets.
Please consult each upstream project for the full license text and terms.

| Asset | Use in this submission | License |
| --- | --- | --- |
| Brax | Continuous-control benchmark environments | Apache-2.0 |
| POPGym | Partially observable benchmark tasks | MIT |
| Mamba / `mamba-ssm` | Base Mamba implementation; `mamba2.py` and `ssd_combined.py` are patched for TiME | Apache-2.0 |
| PyTorch / torchvision / torchaudio | Neural network implementation and training stack | BSD-style |
| JAX / JAXlib | Brax runtime and accelerated numerical backend | Apache-2.0 |
| Gym / Gymnasium | Environment API compatibility | MIT |

The patched Mamba files retain upstream copyright notices where present. No proprietary datasets, pretrained models, or restricted-use assets are redistributed with this supplemental package.
