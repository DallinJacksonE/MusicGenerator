import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
import math
import glob
import os
from midi_utils import sequence_to_midi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch.nn.modules.linear")

# ---------------------------------------------------------
# PyTorch Dataset (Updated for Pre-Parsed Tensors)
# ---------------------------------------------------------


class MIDIDataset(Dataset):
    def __init__(self, tensor_dir):
        # Recursively find all .pt files in the target directory
        self.file_paths = glob.glob(os.path.join(tensor_dir, "*.pt"))
        if not self.file_paths:
            print(f"WARNING: No tensors found in {
                  tensor_dir}. Did you run preprocess.py?")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the tensor instantly from the NVMe drive
        sequence = torch.load(self.file_paths[idx], weights_only=True)

        x = sequence[:-1]
        y = sequence[1:]
        return x, y

# ---------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MIDITransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super(MIDITransformer, self).__init__()
        self.d_model = d_model

        self.pitch_embed = nn.Embedding(128, d_model)
        self.inst_embed = nn.Embedding(128, d_model)
        self.vel_proj = nn.Linear(1, d_model)
        self.dur_proj = nn.Linear(1, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model, dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        self.pitch_head = nn.Linear(d_model, 128)
        self.inst_head = nn.Linear(d_model, 128)
        self.vel_head = nn.Linear(d_model, 1)
        self.dur_head = nn.Linear(d_model, 1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        batch_size, seq_len, _ = src.shape

        pitch = src[:, :, 0].long()
        velocity = src[:, :, 1].unsqueeze(-1)
        duration = src[:, :, 2].unsqueeze(-1)
        instrument = src[:, :, 3].long()

        x = self.pitch_embed(pitch) + \
            self.inst_embed(instrument) + \
            self.vel_proj(velocity) + \
            self.dur_proj(duration)

        x = self.pos_encoder(x)
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(x, mask=mask, is_causal=True)

        return {
            'pitch': self.pitch_head(output),
            'velocity': self.vel_head(output).squeeze(-1),
            'duration': self.dur_head(output).squeeze(-1),
            'instrument': self.inst_head(output)
        }

# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------


def generate_music(model, start_note, device, num_notes=100, temperature=1.0):
    model.eval()
    generated_sequence = start_note.clone().to(device)

    with torch.no_grad():
        for _ in range(num_notes):
            preds = model(generated_sequence)

            last_pitch_logits = preds['pitch'][:, -1, :] / temperature
            last_inst_logits = preds['instrument'][:, -1, :] / temperature
            last_vel = preds['velocity'][:, -1].unsqueeze(-1)
            last_dur = preds['duration'][:, -1].unsqueeze(-1)

            pitch_probs = F.softmax(last_pitch_logits, dim=-1)
            next_pitch = torch.multinomial(pitch_probs, num_samples=1)

            inst_probs = F.softmax(last_inst_logits, dim=-1)
            next_inst = torch.multinomial(inst_probs, num_samples=1)

            next_vel = torch.clamp(last_vel, min=0, max=127)
            next_dur = torch.clamp(last_dur, min=0.05, max=5.0)

            next_note = torch.cat(
                [next_pitch.float(), next_vel, next_dur, next_inst.float()], dim=-1).unsqueeze(1)
            generated_sequence = torch.cat(
                [generated_sequence, next_note], dim=1)

    return generated_sequence.squeeze(0)


# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":

    # Execution Flags (Toggle these to run specific phases)
    RUN_PHASE_1 = False
    RUN_PHASE_2 = False
    RUN_PHASE_3 = False

    # Hardware Tuning Parameters
    OPT_BATCH_SIZE = 32  # Massive chunk for 8GB VRAM
    OPT_WORKERS = 8        # Fast NVMe/CPU loading

    # Check Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")

    # Initialize Model & Loss Functions
    model = MIDITransformer(d_model=128, nhead=4, num_layers=4).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # =====================================================================
    # PHASE 1: General Music Pre-Training
    # =====================================================================
    if RUN_PHASE_1:
        print("\n" + "="*50)
        print("PHASE 1: GENERAL MUSIC PRE-TRAINING")
        print("="*50)

        phase1_dataset = MIDIDataset("tensor_data/phase1_general")
        print(f"[INFO] Successfully loaded {
              len(phase1_dataset)} files into Phase 1 Dataset.")

        phase1_dataloader = DataLoader(
            phase1_dataset,
            batch_size=OPT_BATCH_SIZE,
            shuffle=True,
            num_workers=OPT_WORKERS,
            pin_memory=True,
            prefetch_factor=2
        )
        print(f"[INFO] Batches per epoch: {
              len(phase1_dataloader)} (Batch Size: {OPT_BATCH_SIZE})")

        optimizer_p1 = optim.Adam(model.parameters(), lr=1e-4)
        num_epochs_p1 = 30
        checkpoint_dir1 = 'checkpoints/phase1'
        os.makedirs(checkpoint_dir1, exist_ok=True)

        epoch_losses = []

        print(f"\nStarting Phase 1 Training ({num_epochs_p1} Epochs)...")
        for epoch in range(num_epochs_p1):
            model.train()
            total_loss = 0.0

            for batch_idx, (x, y) in enumerate(phase1_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer_p1.zero_grad()

                preds = model(x)

                pitch_loss = criterion_ce(
                    preds['pitch'].reshape(-1, 128), y[:, :, 0].long().reshape(-1))
                inst_loss = criterion_ce(
                    preds['instrument'].reshape(-1, 128), y[:, :, 3].long().reshape(-1))
                vel_loss = criterion_mse(preds['velocity'], y[:, :, 1])
                dur_loss = criterion_mse(preds['duration'], y[:, :, 2])

                loss = pitch_loss + inst_loss + vel_loss + dur_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer_p1.step()
                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs_p1}], Step [{batch_idx}/{
                          len(phase1_dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(phase1_dataloader)
            print(
                f"--- Epoch [{epoch+1}/{num_epochs_p1}] Average Loss: {avg_loss:.4f} ---")
            epoch_losses.append(avg_loss)

            cp_path = os.path.join(
                checkpoint_dir1, f'phase1_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), cp_path)
            print(f"Saved checkpoint to {cp_path}")

        torch.save(model.state_dict(), 'phase1_weights.pth')
        print("Saved Phase 1 Final Weights: phase1_weights.pth")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs_p1 + 1), epoch_losses,
                 marker='o', linestyle='-', color='b')
        plt.title('Phase 1 General Music Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig('phase1_loss_plot.png')
        plt.close()
        print("Saved Phase 1 loss plot to phase1_loss_plot.png")

    # =====================================================================
    # PHASE 2: Lofi Fine-Tuning
    # =====================================================================
    if RUN_PHASE_2:
        print("\n" + "="*50)
        print("PHASE 2: LOFI FINE-TUNING")
        print("="*50)

        if not RUN_PHASE_1 and os.path.exists('phase1_weights.pth'):
            print("Loading Phase 1 weights before starting Phase 2...")
            model.load_state_dict(torch.load(
                'phase1_weights.pth', map_location=device))

        phase2_dataset = MIDIDataset("tensor_data/phase2_lofi")
        print(f"[INFO] Successfully loaded {
              len(phase2_dataset)} files into Phase 2 Dataset.")

        phase2_dataloader = DataLoader(
            phase2_dataset,
            batch_size=OPT_BATCH_SIZE,
            shuffle=True,
            num_workers=OPT_WORKERS,
            pin_memory=True,
            prefetch_factor=2
        )
        print(f"[INFO] Batches per epoch: {
              len(phase2_dataloader)} (Batch Size: {OPT_BATCH_SIZE})")

        optimizer_p2 = optim.Adam(model.parameters(), lr=1e-5)
        num_epochs_p2 = 50
        checkpoint_dir2 = 'checkpoints/phase2'
        os.makedirs(checkpoint_dir2, exist_ok=True)
        epoch_losses_2 = []

        print(f"\nStarting Phase 2 Fine-Tuning ({num_epochs_p2} Epochs)...")
        for epoch in range(num_epochs_p2):
            model.train()
            total_loss = 0.0

            for batch_idx, (x, y) in enumerate(phase2_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer_p2.zero_grad()

                preds = model(x)

                pitch_loss = criterion_ce(
                    preds['pitch'].reshape(-1, 128), y[:, :, 0].long().reshape(-1))
                inst_loss = criterion_ce(
                    preds['instrument'].reshape(-1, 128), y[:, :, 3].long().reshape(-1))
                vel_loss = criterion_mse(preds['velocity'], y[:, :, 1])
                dur_loss = criterion_mse(preds['duration'], y[:, :, 2])

                loss = pitch_loss + inst_loss + vel_loss + dur_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer_p2.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(phase2_dataloader)
            print(
                f"--- Epoch [{epoch+1}/{num_epochs_p2}] Average Loss: {avg_loss:.4f} ---")
            epoch_losses_2.append(avg_loss)

            cp_path = os.path.join(
                checkpoint_dir2, f'phase2_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), cp_path)
            print(f"Saved checkpoint to {cp_path}")

        torch.save(model.state_dict(), 'phase2_weights.pth')
        print("Saved Phase 2 Final Weights: phase2_weights.pth")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs_p2 + 1), epoch_losses_2,
                 marker='o', linestyle='-', color='g')
        plt.title('Phase 2 Lofi Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig('phase2_loss_plot.png')
        plt.close()

    # =====================================================================
    # PHASE 3: Secondary Genre Fine-Tuning
    # =====================================================================
    if RUN_PHASE_3:
        print("\n" + "="*50)
        print("PHASE 3: SECONDARY GENRE FINE-TUNING")
        print("="*50)

        if not RUN_PHASE_2 and os.path.exists('phase2_weights.pth'):
            print("Loading Phase 2 weights before starting Phase 3...")
            model.load_state_dict(torch.load(
                'phase2_weights.pth', map_location=device))

        phase3_dataset = MIDIDataset("tensor_data/phase3_secondary")
        print(f"[INFO] Successfully loaded {
              len(phase3_dataset)} files into Phase 3 Dataset.")

        phase3_dataloader = DataLoader(
            phase3_dataset,
            batch_size=OPT_BATCH_SIZE,
            shuffle=True,
            num_workers=OPT_WORKERS,
            pin_memory=True,
            prefetch_factor=2
        )
        print(f"[INFO] Batches per epoch: {
              len(phase3_dataloader)} (Batch Size: {OPT_BATCH_SIZE})")

        optimizer_p3 = optim.Adam(model.parameters(), lr=1e-5)
        num_epochs_p3 = 10
        checkpoint_dir3 = 'checkpoints/phase3'
        os.makedirs(checkpoint_dir3, exist_ok=True)
        epoch_losses_3 = []

        print(f"\nStarting Phase 3 Fine-Tuning ({num_epochs_p3} Epochs)...")
        for epoch in range(num_epochs_p3):
            model.train()
            total_loss = 0.0

            for batch_idx, (x, y) in enumerate(phase3_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer_p3.zero_grad()

                preds = model(x)

                pitch_loss = criterion_ce(
                    preds['pitch'].reshape(-1, 128), y[:, :, 0].long().reshape(-1))
                inst_loss = criterion_ce(
                    preds['instrument'].reshape(-1, 128), y[:, :, 3].long().reshape(-1))
                vel_loss = criterion_mse(preds['velocity'], y[:, :, 1])
                dur_loss = criterion_mse(preds['duration'], y[:, :, 2])

                loss = pitch_loss + inst_loss + vel_loss + dur_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer_p3.step()
                total_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs_p3}], Step [{batch_idx}/{
                          len(phase3_dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(phase3_dataloader)
            print(
                f"--- Epoch [{epoch+1}/{num_epochs_p3}] Average Loss: {avg_loss:.4f} ---")
            epoch_losses_3.append(avg_loss)

            cp_path = os.path.join(
                checkpoint_dir3, f'phase3_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), cp_path)
            print(f"Saved checkpoint to {cp_path}")

        torch.save(model.state_dict(), 'phase3_weights.pth')
        print("Saved Phase 3 Final Weights: phase3_weights.pth")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs_p3 + 1), epoch_losses_3,
                 marker='o', linestyle='-', color='r')
        plt.title('Phase 3 Secondary Genre Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig('phase3_loss_plot.png')
        plt.close()

# =====================================================================
    # POST-TRAINING GENERATION
    # =====================================================================
    print("\n" + "="*50)
    print("FINAL MODEL GENERATION (LOFI)")
    print("="*50)

    # Explicitly load the Phase 2 (LoFi) weights
    if os.path.exists('phase2_weights.pth'):
        print("Loading Phase 2 LoFi weights...")
        model.load_state_dict(torch.load(
            'phase2_weights.pth', map_location=device))
    else:
        print(
            "WARNING: phase2_weights.pth not found. Generating with whatever is in memory.")

    # Seed the model with a Middle C on an Acoustic Grand Piano
    start_note = torch.tensor([[[60.0, 80.0, 0.5, 0.0]]], dtype=torch.float32)

    print("Generating pure LoFi sequence (150 notes)...")

    # Dropped temperature from 1.2 to 0.85 for musical coherence
    generated_seq = generate_music(
        model, start_note, device, num_notes=150, temperature=0.85)

    output_name = "pure_lofi_output.mid"
    sequence_to_midi(generated_seq, output_filename=output_name)
    print(f"\nGeneration complete! Check {output_name} to hear the results.")
