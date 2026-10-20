@echo off
REM 环境初始化:PyPI 走阿里云镜像,torch/torchvision 的 CUDA 版走阿里云 pytorch-wheels
py -m venv .venv
.venv\Scripts\python -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple/
.venv\Scripts\pip install numpy pandas scikit-learn matplotlib seaborn pillow gradio kagglehub tqdm -i https://mirrors.aliyun.com/pypi/simple/
.venv\Scripts\pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 -f https://mirrors.aliyun.com/pytorch-wheels/cu130/ -i https://mirrors.aliyun.com/pypi/simple/
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
pause
