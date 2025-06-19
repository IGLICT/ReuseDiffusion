## 简介
基于时序冗余重用的扩散模型推理加速方案，在保证生成质量的前提下实现10%以上的推理时延减少。
支持模型：CogVideoX，SDXL、SD3、FLUX。
支持加速器类型：NVIDIA A100/800 H100/800 GPU

## 安装
### 前置条件
```
# Change cu126 to your CUDA version
pip3 install --index-url https://download.pytorch.org/whl/cu126 \
    'torch>=2.4.0' 'triton>=3.1.0' 'diffusers>=0.32.2'
```
注：未在此前版本测试过。
### 下载
```
git clone https://github.com/DongBaiYue/ReuseDiffusion.git
```


## 用法

### 下载模型参数
运行模型前需要首先下载模型参数，以cogvideox为例
```
cd ReuseDiffusion
# 下载模型，模型参数保存到 ./models/cogvideox。若已下载，跳转到下一步测试。
python3 down_models.py --mode download --models cogvideox --save_dir ./models
# 测试模型，--save_dir指定模型参数存储位置，生成内容为 ./results/cogvideox/test.mp4
python3 down_models.py --mode test --models cogvideox --save_dir ./models
```
### 运行模型
```
# original
python3 generate.py --model cogvideox --model_dir ./models --mode original --benchmark --gpu 0
# reuse，优化版
python3 generate.py --model cogvideox --model_dir ./models --mode reuse --benchmark --gpu 0
```
generate.py包含更多可选参数，参数含义参考generate.py。
- prompts_num：指定输入含有多少个prompts，默认为1
- batch_size：一次执行多少个prompts的生成过程，默认为1。比如prompts_num=8，batch_size=4，将会执行2个batch。
- steps：扩散模型去噪步数，默认为28。指定steps为其它值时，第一次运行需要收集统计数据。
- collect_diff：收集统计数据。
- max_skip_steps，threshold：跳过决策的超参数。
- benchmark：测量执行时间。

## 代码结构
ReuseDiffusion
- attn_processor       attn、layernorm优化实现
- input                prompt和收集的统计数据
- pipelines            是否reuse的决策函数，及对4种模型pipeline的修改。
- results              生成结果目录
- down_models.py       用于下载、测试模型
- generate.py          主文件，
- util.py              废弃，推理框架相关（sdfast）

## 性能

| 模型 | original(s) | reuse(s) |
|---------|---------|---------|
| cogvideox | 68.43  | 60.94  |
| flux | 13.41  | 11.99  |
| sd3 | 5.04  | 4.22  |
| sdxl | 3.08  | 2.94  |

## 本文方法及相关工作
- [DiTFastAttn](https://openreview.net/forum?id=51HQpkQy3t) 时间稀疏注意力。
- [Sparse VideoGen](https://arxiv.org/abs/2502.01776) 空间-时间稀疏注意力。
- [AdapativeDiffusion](https://arxiv.org/abs/2410.09873) 动态差分决策。
- [本文方法] 基于同一扩散模型在不同Prompt下差分的稳定性，静态-动态差分决策；同时优化LayerNorm。

## 贡献
中国科学院计算技术研究所高性能计算机研究中心系统软件组  

dtk@ncic.ac.cn
