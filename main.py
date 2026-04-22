import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import warnings

from dataset import MIDIDataset
from model import MIDITransformer
from generation import generate_music
from midi_utils import sequence_to_midi
from trainer import train_phase
from genre_config import BASE_CONFIG, GENRE_REGISTRY

warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch.nn.modules.linear")

if __name__ == "__main__":

    # --- PIPELINE CONTROLS ---
    RUN_PRETRAIN = True      # Phase 1: General Music
    RUN_FINETUNE = True       # Phase 2: Targeted Genre

    # Select your genre right here!
    # Must match a key in GENRE_REGISTRY (e.g., "lofi", "metal")
    ACTIVE_GENRE = "lofi"

    OPT_BATCH_SIZE = 32
    OPT_WORKERS = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")
    model = MIDITransformer(d_model=256, nhead=8, num_layers=6).to(device)

    # =====================================================================
    # PHASE 1: General Music Pre-Training
    # =====================================================================
    if RUN_PRETRAIN:
        print("\n" + "="*50 + "\nPHASE 1: GENERAL MUSIC PRE-TRAINING\n")
        print("="*50)
        full_dataset = MIDIDataset(BASE_CONFIG["tensor_dir"])

        # Calculate split sizes (90% Train, 10% Validation)
        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset,
                                  batch_size=OPT_BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=OPT_WORKERS,
                                  pin_memory=True,
                                  prefetch_factor=2)
        val_loader = DataLoader(val_dataset,
                                batch_size=OPT_BATCH_SIZE,
                                shuffle=False,
                                num_workers=OPT_WORKERS,
                                pin_memory=True,
                                prefetch_factor=2)

        optimizer = optim.Adam(model.parameters(), lr=BASE_CONFIG["lr"])

        train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=BASE_CONFIG["epochs"],
            device=device,
            phase_name=BASE_CONFIG["name"],
            checkpoint_dir='checkpoints/phase1',
            weights_save_path=BASE_CONFIG["weights_file"],
            plot_save_path='phase1_loss_plot.png'
        )

    # =====================================================================
    # PHASE 2: Dynamic Genre Fine-Tuning
    # =====================================================================
    if RUN_FINETUNE:
        if ACTIVE_GENRE not in GENRE_REGISTRY:
            raise ValueError(f"Genre '{ACTIVE_GENRE}' not found in registry!")

        genre_data = GENRE_REGISTRY[ACTIVE_GENRE]
        genre_name = ACTIVE_GENRE.capitalize()

        print("\n" + "="*50 +
              f"\nPHASE 2: {genre_name.upper()} FINE-TUNING\n" + "="*50)

        # Always start fine-tuning from the base general model
        if not RUN_PRETRAIN and os.path.exists(BASE_CONFIG["weights_file"]):
            print(f"Loading Base {BASE_CONFIG['name']} weights...")
            model.load_state_dict(torch.load(
                BASE_CONFIG["weights_file"],
                map_location=device,
                weights_only=True))

        full_dataset = MIDIDataset(genre_data["tensor_dir"])
        total_files = len(full_dataset)
        print(f"[INFO] Loaded {total_files} tensors for {ACTIVE_GENRE}.")

        if total_files == 0:
            raise RuntimeError(f"0 files found for "
                               f"{ACTIVE_GENRE}! Lower "
                               f"SEQ_LENGTH in preprocess.py.")
        elif total_files == 1:
            print(
                f"[WARNING] Only 1 file found. "
                f"Bypassing split to prevent DataLoader crash.")
            train_dataset = full_dataset
            val_dataset = full_dataset
        else:
            # Calculate split sizes safely
            val_size = max(1, int(0.1 * total_files))
            train_size = total_files - val_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size])
        # Calculate split sizes (90% Train, 10% Validation)
        # Ensure at least 1 file in validation if dataset is tiny
        val_size = max(1, int(0.1 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=OPT_BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=OPT_WORKERS,
                                  pin_memory=True,
                                  prefetch_factor=2)
        val_loader = DataLoader(val_dataset,
                                batch_size=OPT_BATCH_SIZE,
                                shuffle=False,
                                num_workers=OPT_WORKERS,
                                pin_memory=True,
                                prefetch_factor=2)

        optimizer = optim.Adam(model.parameters(), lr=genre_data["lr"])
        weights_out = f"genre_{ACTIVE_GENRE}_weights.pth"

        train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=genre_data["epochs"],
            device=device, phase_name=f"{genre_name} Fine-Tuning",
            checkpoint_dir=f'checkpoints/genre_{ACTIVE_GENRE}',
            weights_save_path=weights_out,
            plot_save_path=f'{ACTIVE_GENRE}_loss_plot.png'
        )

    # =====================================================================
    # POST-TRAINING GENERATION
    # =====================================================================
    genre_name = ACTIVE_GENRE.capitalize()
    print("\n" + "="*50 +
          f"\nFINAL MODEL GENERATION ({genre_name.upper()})\n" + "="*50)

    target_weights = f"genre_{ACTIVE_GENRE}_weights.pth"
    if os.path.exists(target_weights):
        model.load_state_dict(torch.load(target_weights,
                                         map_location=device,
                                         weights_only=True))

    # Grab the allowed instruments directly from the registry
    allowed_insts = GENRE_REGISTRY[ACTIVE_GENRE].get(
        "allowed_instruments", None)

    start_note = torch.tensor(
        [[[60.0, 80.0, 0.5, 0.0, 0.0]]], dtype=torch.float32)
    print(f"Generating pure {genre_name} sequence (150 notes)...")

    # Update generation.py to accept the allowed_insts list dynamically!
    generated_seq = generate_music(
        model,
        start_note,
        device,
        allowed_insts,
        num_notes=150,
        temperature=0.85,
    )

    output_name = f"pure_{ACTIVE_GENRE}_output.mid"
    sequence_to_midi(generated_seq, output_filename=output_name)
    print(f"\nGeneration complete! Check {output_name} to hear the results.")
