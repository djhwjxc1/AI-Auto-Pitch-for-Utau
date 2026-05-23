"""
train_vae_tcn.py - TCN + VAE 音高生成模型 (~100万参数)
"""

import os, json, hashlib, numpy as np, torch, torch.nn as nn, librosa
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from torchfcpe import spawn_bundled_infer_model

# ==================== F0 提取 ====================
class F0Extractor:
    _model = None
    @staticmethod
    def extract(wav_path, frame_ms=10.0):
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
            y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-8)
            if len(y) < sr * 0.1: return None
            if F0Extractor._model is None:
                F0Extractor._model = spawn_bundled_infer_model(device='cuda' if torch.cuda.is_available() else 'cpu')
            audio = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(-1)
            if torch.cuda.is_available(): audio = audio.cuda()
            hop = int(sr * frame_ms / 1000)
            f0 = F0Extractor._model.infer(audio, sr=sr, decoder_mode='local_argmax', threshold=0.006,
                f0_min=50, f0_max=2000, interp_uv=False, output_interp_target_length=(len(y)//hop)+1)
            f0_log = np.where(f0.cpu().numpy().flatten() > 0, np.log(np.clip(f0.cpu().numpy().flatten(), 1e-10, None)), -10.0)
            return {'f0_log': f0_log, 'duration_ms': len(y)/sr*1000}
        except Exception as e:
            print(f"F0 提取失败: {e}"); return None

# ==================== 标签生成 ====================
class AutoLabelGenerator:
    def __init__(self, frame_ms=10.0):
        self.frame_ms, self.extractor = frame_ms, F0Extractor()
    def process_wav(self, wav_path, output_dir="auto_labels"):
        os.makedirs(output_dir, exist_ok=True)
        data = self.extractor.extract(wav_path)
        if data is None: return []
        f0_log = data['f0_log']
        min_f, max_f = int(150/self.frame_ms), int(500/self.frame_ms)
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        hl = int(sr*self.frame_ms/1000)
        rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=hl)[0][:len(f0_log)], ref=np.max)
        silence = np.where(np.diff((rms_db<-25).astype(int))==1)[0]
        pitch_bounds = np.where(np.abs(np.diff(f0_log))>0.8)[0]
        bounds = sorted(set([0]+silence.tolist()+pitch_bounds.tolist()+[len(f0_log)]))
        final = [0]
        for i in range(len(bounds)-1):
            s, e = bounds[i], bounds[i+1]
            if e-s > max_f:
                for sub in range(s+min_f, e, min_f): final.append(sub)
            final.append(e)
        final = sorted(set(final))
        notes = []
        for i in range(len(final)-1):
            s, e = final[i], final[i+1]
            if e-s < 3: continue
            vf = f0_log[s:e][f0_log[s:e]>-5.0]
            if len(vf)>2:
                midi = int(round(69+12*np.log2(np.clip(np.exp(np.mean(vf)),20,20000)/440.0)))
                notes.append({'start_frame':s,'end_frame':e,'midi':max(36,min(96,midi)),'duration_frames':e-s})
        samples = []
        for i, n in enumerate(notes):
            prev_m = notes[i-1]['midi'] if i>0 else n['midi']
            next_m = notes[i+1]['midi'] if i<len(notes)-1 else n['midi']
            sample = {'note':n['midi'],'duration_ms':(n['end_frame']-n['start_frame'])*self.frame_ms,'f0_curve':f0_log[n['start_frame']:n['end_frame']].tolist(),'prev_midi':prev_m,'next_midi':next_m}
            samples.append(sample)
            with open(os.path.join(output_dir,f"sample_{hashlib.md5(f'{wav_path.name}_{i}'.encode()).hexdigest()[:8]}.json"),'w') as f: json.dump(sample,f)
        return samples

# ==================== 数据集 ====================
class PitchDataset(Dataset):
    def __init__(self, data_dir, seq_length=50):
        self.seq_length = seq_length
        self.sequences = []
        for f in tqdm(list(Path(data_dir).glob("*.json")), desc="加载中"):
            try:
                s = json.load(open(f))
                note, curve, dur = s['note'], s['f0_curve'], len(s['f0_curve'])
                if dur < 5: continue
                prev_m, next_m = s.get('prev_midi',note), s.get('next_midi',note)
                stride = max(1, seq_length//4)
                for start in range(0, dur, stride):
                    end, al = min(start+seq_length, dur), min(seq_length, dur-start)
                    if al<5: continue
                    feats = np.zeros((seq_length,5), dtype=np.float32)
                    targs = np.zeros(seq_length, dtype=np.float32)
                    for j in range(al):
                        fi = start+j
                        feats[j] = [note/127.0, 1.0 if fi==0 else 0.0, (dur-fi)/max(dur,1), prev_m/127.0, next_m/127.0]
                        ref_log = np.log(440.0*(2.0**((note-69)/12.0)))
                        targs[j] = np.clip((curve[fi]-ref_log)*1730.0/50.0, -1.0, 1.0)
                    self.sequences.append((feats, targs))
            except: continue
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        feats, targs = self.sequences[idx]
        return torch.FloatTensor(feats), torch.FloatTensor(targs)

# ==================== TCN + VAE 模型 (~100万参数) ====================
class PitchVAE(nn.Module):
    def __init__(self, input_size=5, hidden=320, latent=160):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden, 3, padding=1, dilation=1), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, 3, padding=8, dilation=8), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, label=None):
        b, s, _ = x.shape
        h = self.tcn(x.transpose(1, 2))
        h = h[:, :, :s].transpose(1, 2).reshape(-1, h.shape[1])
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        out = self.decoder(z).reshape(b, s)
        if label is not None:
            recon = nn.SmoothL1Loss()(out, label.reshape(b, s))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return out, recon, kl
        return out

# ==================== 训练 ====================
def main():
    LABEL_DIR = "auto_labels"
    if len(list(Path(LABEL_DIR).glob("*.json"))) <= 100:
        gen = AutoLabelGenerator()
        for wav in tqdm([w for w in Path(".").glob("*.wav") if w.stat().st_size>=10000], desc="处理音频"): gen.process_wav(wav, LABEL_DIR)
    dataset = PitchDataset(LABEL_DIR)
    tr_sz = int(0.85*len(dataset))
    tr_set, val_set = torch.utils.data.random_split(dataset, [tr_sz, len(dataset)-tr_sz], generator=torch.Generator().manual_seed(42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs = 128 if device.type=='cuda' else 64
    tr_ld = DataLoader(tr_set, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    val_ld = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0)
    model = PitchVAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    best_val, patience, no_imp = float('inf'), 20, 0
    for epoch in range(100):
        model.train(); tr_loss = 0
        for feats, targs in tqdm(tr_ld, desc="训练中"):
            feats, targs = feats.to(device), targs.to(device)
            opt.zero_grad()
            out, recon, kl = model(feats, targs)
            loss = recon + 0.001 * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)
        model.eval(); val_loss = 0
        with torch.no_grad():
            for feats, targs in val_ld:
                feats, targs = feats.to(device), targs.to(device)
                out, recon, kl = model(feats, targs)
                val_loss += (recon + 0.001 * kl).item()
        val_loss /= len(val_ld)
        print(f"Epoch {epoch+1:3d}/100 | Train: {tr_loss:.6f} | Val: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss; torch.save(model.state_dict(), "pitch_vae_tcn.pth"); print("  ✓ 保存"); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: print(f"⚠ 早停于 {epoch+1}"); break
    model.load_state_dict(torch.load("pitch_vae_tcn.pth"))
    model.eval()
    torch.onnx.export(model, torch.randn(1,50,5).to(device), "pitch_vae_tcn.onnx",
        input_names=['input_features'], output_names=['output'],
        dynamic_axes={'input_features':{0:'batch',1:'sequence'},'output':{0:'batch',1:'sequence'}},
        opset_version=17, dynamo=False)
    print("✅ ONNX: pitch_vae_tcn.onnx")

if __name__ == "__main__":
    main()