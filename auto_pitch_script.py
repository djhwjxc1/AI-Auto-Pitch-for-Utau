"""
auto_pitch.py - OpenUtau 自动音高插件
接收临时 UST 文件，生成音高曲线后写回
"""

import sys
import os
import numpy as np
import onnxruntime as ort


def parse_ust(ust_path):
    notes = []
    current = {}
    with open(ust_path, 'rb') as f:
        raw = f.read()
    raw = raw.replace(b'\r\n', b'\n')

    for line in raw.split(b'\n'):
        try:
            line = line.decode('shift_jis').strip()
        except:
            line = line.decode('utf-8').strip()

        if line.startswith('[#') and current:
            if 'NoteNum' in current:
                notes.append(current)
            current = {}
        elif '=' in line:
            key, value = line.split('=', 1)
            current[key] = value
    if current and 'NoteNum' in current:
        notes.append(current)
    return notes


def write_ust(ust_path, notes):
    with open(ust_path, 'rb') as f:
        raw = f.read()

    raw = raw.replace(b'\r\n', b'\n')
    lines = raw.split(b'\n')

    note_idx = 0
    result = b''
    skip_old = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(b'[#') and len(stripped) > 3 and stripped[2:3].isdigit():
            skip_old = False
            result += line + b'\n'

            if note_idx < len(notes):
                n = notes[note_idx]
                if 'PBS' in n:
                    result += f"PBS={n['PBS']}\n".encode('utf-8')
                if 'PBW' in n:
                    result += f"PBW={n['PBW']}\n".encode('utf-8')
                if 'PBY' in n:
                    result += f"PBY={n['PBY']}\n".encode('utf-8')
                if 'PBM' in n:
                    result += f"PBM={n['PBM']}\n".encode('utf-8')
                skip_old = True
            note_idx += 1
            continue

        if skip_old and (stripped.startswith(b'PBS') or stripped.startswith(b'PBW') or
                         stripped.startswith(b'PBY') or stripped.startswith(b'PBM')):
            continue

        result += line + b'\n'

    with open(ust_path, 'wb') as f:
        f.write(result)


def generate_pitch(model_path, notes):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.log')

    with open(log_path, 'a') as f:
        f.write(f"加载模型: {model_path}\n")

    session = ort.InferenceSession(model_path)

    with open(log_path, 'a') as f:
        f.write("模型加载成功\n")

    features = []
    note_info = []

    for i, note in enumerate(notes):
        note_num = int(note.get('NoteNum', 60))
        length = int(note.get('Length', 480))
        tempo = float(note.get('Tempo', 120))
        duration_ms = length * 60000 / (tempo * 480)
        frames = max(1, int(duration_ms / 10))
        note_info.append((note_num, frames, duration_ms))

        with open(log_path, 'a') as f:
            f.write(f"音符{i}: NoteNum={note_num}, Length={length}, Tempo={tempo}, duration_ms={duration_ms:.1f}, frames={frames}\n")

        prev_num = int(notes[i - 1]['NoteNum']) if i > 0 else note_num
        next_num = int(notes[i + 1]['NoteNum']) if i < len(notes) - 1 else note_num

        for f in range(frames):
            features.extend([
                note_num / 127.0,
                1.0 if f == 0 else 0.0,
                (frames - f) / frames,
                prev_num / 127.0,
                next_num / 127.0
            ])

    total_frames = len(features) // 5
    segment_len = 50
    all_outputs = []

    for start in range(0, total_frames, segment_len):
        end = min(start + segment_len, total_frames)
        chunk_len = end - start

        chunk = features[start * 5:end * 5]
        if chunk_len < segment_len:
            chunk.extend([0.0] * ((segment_len - chunk_len) * 5))

        input_tensor = np.array(chunk, dtype=np.float32).reshape(1, segment_len, 5)
        out = session.run(None, {'input_features': input_tensor})
        out = out[0].flatten()

        if chunk_len < segment_len:
            out = out[:chunk_len]

        all_outputs.append(out)

    pitch_output = np.concatenate(all_outputs)
    offset = 0

    for i, (note_num, frames, duration_ms) in enumerate(note_info):
        pitch_points = []
        for f in range(frames):
            idx = offset + f
            if idx >= len(pitch_output):
                break

            cents = float(pitch_output[idx]) * 2
            cents = max(-30,min(60, cents))
            pitch_points.append(int(cents * 10))

        if pitch_points:
            n = len(pitch_points)
            notes[i]['PBS'] = f'-40;{pitch_points[0]}'

            if n > 1:
                # 控制点均匀分布在音符时长内
                total_time = frames * 10  # 音符总时长(ms)
                avg_interval = int(total_time / (n - 1))  # 平均间隔
                notes[i]['PBW'] = ','.join([str(avg_interval)] * (n - 1))
                notes[i]['PBY'] = ','.join(str(v) for v in pitch_points[1:])
                notes[i]['PBM'] = ','.join(['s'] * (n - 1))

            with open(log_path, 'a') as f:
                f.write(f"音符{i} PBY前10: {pitch_points[:10]}\n")

        offset += frames

    return notes


def main():
    if getattr(sys, 'frozen', False):
        plugin_dir = os.path.dirname(sys.executable)
    else:
        plugin_dir = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(plugin_dir, 'error.log')
    with open(log_path, 'w') as f:
        f.write(f"收到参数: {sys.argv}\n")

    if len(sys.argv) < 2:
        with open(log_path, 'a') as f:
            f.write("❌ 没有收到 UST 文件路径\n")
        return

    ust_path = sys.argv[1]
    model_path = os.path.join(plugin_dir, 'pitch_model.onnx')

    with open(log_path, 'a') as f:
        f.write("开始解析 UST...\n")

    notes = parse_ust(ust_path)

    with open(log_path, 'a') as f:
        f.write(f"解析到 {len(notes)} 个音符\n")

    if len(notes) == 0:
        with open(log_path, 'a') as f:
            f.write("❌ 没有音符，退出\n")
        return

    with open(log_path, 'a') as f:
        f.write("开始生成音高...\n")

    notes = generate_pitch(model_path, notes)

    with open(log_path, 'a') as f:
        f.write("生成完成，开始写回...\n")

    write_ust(ust_path, notes)

    with open(log_path, 'a') as f:
        f.write("✅ 完成\n")


if __name__ == '__main__':
    main()