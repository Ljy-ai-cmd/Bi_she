@echo off
chcp 65001 >nul
echo ****************** Installing pytorch ******************
:: 使用清华镜像加速下载 CUDA 11.8 版本
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo.
echo ****************** Installing yaml ******************
pip install PyYAML

echo.
echo.
echo ****************** Installing easydict ******************
pip install easydict

echo.
echo.
echo ****************** Installing cython ******************
pip install cython

echo.
echo.
echo ****************** Installing opencv-python ******************
pip install opencv-python

echo.
echo.
echo ****************** Installing pandas ******************
pip install pandas

echo.
echo.
echo ****************** Installing tqdm ******************
pip install tqdm

echo.
echo.
echo ****************** Installing coco toolkit ******************
pip install pycocotools-windows -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo.
echo ****************** Installing jpeg4py python wrapper ******************
pip install jpeg4py

echo.
echo.
echo ****************** Installing tensorboard ******************
pip install tb-nightly

echo.
echo.
echo ****************** Installing tikzplotlib ******************
pip install tikzplotlib

echo.
echo.
echo ****************** Installing thop tool for FLOPs and Params computing ******************
pip install thop -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo.
echo ****************** Installing colorama ******************
pip install colorama

echo.
echo.
echo ****************** Installing lmdb ******************
pip install lmdb

echo.
echo.
echo ****************** Installing scipy ******************
pip install scipy

echo.
echo.
echo ****************** Installing visdom ******************
pip install visdom

echo.
echo.
echo ****************** Installing vot-toolkit python ******************
pip install vot-toolkit -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo.
echo ****************** Installing timm ******************
pip install timm==0.5.4

echo.
echo.
echo ****************** Installing yacs ******************
pip install yacs

echo.
echo.
echo ****************** Installing pytorch-pretrained-bert ******************
pip install pytorch-pretrained-bert==0.6.2

echo.
echo.
echo ****************** Installing scikit-image ******************
pip install scikit-image

echo.
echo.
echo ****************** Installing CLIP ******************
pip install ftfy regex tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install git+https://github.com/openai/CLIP.git

echo.
echo.
echo ****************** Installation complete! ******************
pause