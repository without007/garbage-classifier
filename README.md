# 生活垃圾智能分类系统

基于 CNN 的生活垃圾五分类(塑料 / 金属/易拉罐 / 玻璃 / 纸张/纸盒 / 厨余)课程设计项目。
数据集:Kaggle [mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)(12 类合并映射为 5 类,8:1:1 分层划分)。

## 1. 环境(只需一次)

双击 `setup_env.bat`,或在终端执行:

```bat
py -m venv .venv
.venv\Scripts\pip install -U pip -i https://mirrors.aliyun.com/pypi/simple/
.venv\Scripts\pip install numpy pandas scikit-learn matplotlib seaborn pillow gradio kagglehub tqdm -i https://mirrors.aliyun.com/pypi/simple/
```

> torch CUDA 版走阿里云镜像(清华镜像缺 Windows CUDA 包,且本网络下不可用):
> ```bat
> .venv\Scripts\pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 -f https://mirrors.aliyun.com/pytorch-wheels/cu130/
> ```
> 若下载过慢,可用浏览器下载下面两个文件放进 `wheels\`,再执行
> `.venv\Scripts\pip install wheels\*.whl`:
> - `https://mirrors.aliyun.com/pytorch-wheels/cu130/torch-2.14.1%2Bcu130-cp314-cp314-win_amd64.whl`
> - `https://mirrors.aliyun.com/pytorch-wheels/cu130/torchvision-0.29.1%2Bcu130-cp314-cp314-win_amd64.whl`

验证 GPU:

```bat
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

## 2. 获取数据集(二选一)

- **手动(推荐)**:浏览器开代理,从上面的 Kaggle 页面下载 zip,解压到 `data/raw/`(解压后应能看到 plastic、green-glass、biological 等 12 个子目录)。
- **自动(需代理)**:`.venv\Scripts\python scripts\prepare_data.py --download`

然后整理 + 划分(不需要网络):

```bat
.venv\Scripts\python scripts\prepare_data.py
```

## 3. 训练与评估

```bat
:: 基础组:自建 3 层 CNN
.venv\Scripts\python -m src.train --model cnn --weighted --epochs 60

:: 对比组 A:ResNet-18 仅训练分类头(冻结骨干)
.venv\Scripts\python -m src.train --model resnet18 --weighted --epochs 30

:: 对比组 B:ResNet-18 解冻 layer4 微调(精度最高)
.venv\Scripts\python -m src.train --model resnet18 --unfreeze --epochs 20 --tag unfrozen

:: 测试集评估 + 混淆矩阵(标签需与训练时的 --tag 对应)
.venv\Scripts\python -m src.evaluate --model resnet18 --tag unfrozen

:: 三组实验结果汇总对比
.venv\Scripts\python scripts\compare_models.py
```

> `--tag` 会把标签追加到产物文件名,不同实验互不覆盖;省略则用默认名。
> `--weighted` 按类别频率加权损失;`--unfreeze` 解冻 ResNet-18 的 layer4。

产物:`outputs/checkpoints/best_*.pth`、`outputs/figures/curves_*.png`、`outputs/figures/confusion_*.png`、`outputs/figures/compare_models.png`、`outputs/logs/`。

## 4. 演示

```bat
.venv\Scripts\python -m src.app
```

默认加载精度最高的 `best_resnet18_unfrozen.pth`,浏览器打开 http://127.0.0.1:7860 ,支持上传图片或摄像头拍照,显示 Top-3 类别与置信度。加载其他权重用 `--model cnn --tag ""` 之类的方式指定。

## 目录结构

```
src/         config.py 配置 | data.py 数据管道 | models.py 模型
             train.py 训练 | evaluate.py 评估 | app.py Gradio 界面
scripts/     prepare_data.py 数据整理与划分
data/raw     原始数据   data/processed 划分后数据
outputs/     checkpoints 权重 | figures 图表 | logs 日志
```
