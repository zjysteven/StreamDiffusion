# StreamDiffusion with ControlNet Integration

For original README, please refer to [README_original.md](./README_original.md).

## Setup

```
conda env create -f environment.yml
conda activate vc
python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt
cd streamdiffusion
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```