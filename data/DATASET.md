# 当前分析的数据准备

最终分析使用 MER-PS 开发数据及相应的刺激、信号和表征缓存。
原始数据、下载缓存、特征数组与账户令牌均不纳入版本控制。

## 目录约定

```text
data/
├── MER_PS_trainval/           24 名开发集参与者的授权数据
├── feature_cache/             特征缓存
├── innovation_cache/          信号与索引缓存
├── innovation_cache_antialias/ 抗混叠信号缓存
├── stimuli/                   刺激身份审计及内容特征
└── download/                  本地下载缓存
```

具体输入路径以各脚本参数为准。刺激清单位于 `configs/stimuli/refed_15_videos.csv`。

## 下载开发数据

数据来源：[MER-PS train/validation dataset](https://huggingface.co/datasets/MER-PS/MER-PS-trainval)。
账户获准访问后运行：

```bash
bash scripts/download_data.sh
```

下载完成后，可先检查压缩包：

```bash
unzip -tq data/download/MER_PS_trainval.zip
```

解压后确保数据根目录为 `data/MER_PS_trainval/`。

## 准备当前分析输入

信号缓存、CBraMod 表征、刺激审计和内容特征各有独立入口，见[脚本索引](../scripts/README.md)。
依据目标分析准备相应缓存，保持参与者、视频和时间索引一致。
模型来源、数据访问和媒体分享范围应按提供方条件核对。
