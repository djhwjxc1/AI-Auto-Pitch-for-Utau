# AI-Auto-Pitch-for-OpenUtau

选中音符，一键生成自然的颤音、滑音和起音效果。基于 VAE 深度学习模型，40 分钟多歌手清唱数据训练，CPU 推理无需显卡。

## 功能
- 多歌手通用
- 100万参数 TCN-VAE 模型
- CPU 推理和GPU推理皆可

## 安装

1. 下载 `AI-Auto-Pitch-for-OpenUtau.zip`
2. 解压到 `OpenUtau/Plugins/`

## 使用

1. 选中音符
2. 右键 → Plugins → AI Auto Pitch
3. 完成

## 技术栈

| 组件 | 技术 |
|------|------|
| 模型 | TCN-VAE |
| F0 提取 | FCPE |
| 数据 | 40分钟多歌手清唱 |
| 推理 | ONNX Runtime |
| 音高 | ±80 音分，S型曲线 |

## 训练

```bash
pip install torch numpy librosa torchfcpe onnx onnxruntime
python train_vae.py
```

将 `pitch_vae.onnx` 改名为 `pitch_model.onnx` 放入插件目录。

## 许可

MIT

