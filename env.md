```bash
conda create -n test python=3.9
conda activate test
python -m pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
cd streamdiffusion/
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
python -m pip install opencv-python
python -m pip install --upgrade diffusers[torch]
python -m pip install peft
```