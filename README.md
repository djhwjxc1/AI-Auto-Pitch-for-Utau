# AI-Auto-Pitch-for-Utau
AI Auto Pitch 为 OpenUtau 提供 AI 驱动的自然音高曲线生成
AI Auto Pitch for OpenUtau
基于 VAE 的自动音高曲线生成插件，为 OpenUtau 提供自然的颤音和滑音效果。

🎵 功能
选中音符 → 右键 → Plugins → AI Auto Pitch

自动生成自然的音高曲线（颤音、滑音、起音下降）

支持多歌手通用

📦 安装
下载 AI-Auto-Pitch-for-Utau.zip

确保目录结构与github一致

🚀 使用
在 OpenUtau 中选中音符

右键 → Plugins → AI Auto Pitch

音高曲线自动生成

🔧 技术细节
模型：MLP-VAE（~18万参数）

输入：MIDI、起音标志、剩余时长、前后音符

输出：每帧音分偏移（±80音分）

F0提取：FCPE

数据：40分钟多歌手清唱

🛠️ 训练自己的模型
bash
pip install torch numpy librosa torchfcpe
python train_vae.py
将生成的 pitch_vae.onnx 重命名为 pitch_model.onnx 放入插件目录。

📝 许可证
MIT

🤝 致谢
OpenUtau

FCPE

PyTorch


