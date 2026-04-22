import os
import torch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from midi_utils import get_midi_files, parse_midi_file, download_kaggle_datasets

# Define where the raw MIDI lives and where the tensors will go
PHASES = {
    "phase1": {
        "raw_dir": "training_data/phase1_general",
        "tensor_dir": "tensor_data/phase1_general"
    },
    "phase2": {
        "raw_dir": "training_data/phase2_lofi",
        "tensor_dir": "tensor_data/phase2_lofi"
    },
    "phase3": {
        "raw_dir": "training_data/phase3_secondary",
        "tensor_dir": "tensor_data/phase3_secondary"
    }
}

SEQ_LENGTH = 50


def process_single_file(args):
    """Parses a single MIDI file and saves it as a PyTorch tensor."""
    file_path, tensor_dir = args

    try:
        notes = parse_midi_file(file_path)

        if not notes or len(notes) < SEQ_LENGTH + 1:
            return False

        sequence = []
        for note in notes[:SEQ_LENGTH + 1]:
            duration = note['end'] - note['start']
            sequence.append([note['pitch'], note['velocity'],
                            duration, note['instrument_program']])

        tensor_data = torch.tensor(sequence, dtype=torch.float32)

        base_name = os.path.basename(file_path).replace(
            '.mid', '').replace('.midi', '')
        safe_name = f"{base_name}_{abs(hash(file_path))}.pt"
        save_path = os.path.join(tensor_dir, safe_name)

        torch.save(tensor_data, save_path)
        return True

    except Exception:
        return False


if __name__ == "__main__":
    print("--- Step 1: Downloading Datasets from Kaggle ---")

    general_datasets = [
        "soumikrakshit/classical-music-midi",
        "kritanjalijain/maestropianomidi",
        "joebeachcapital/guitar-chords-midi-pitches",
        "imsparsh/lakh-midi-clean",
        "hansespinosa2/40000-video-game-midi-files",
        "alexignatov/the-expanded-groove-midi-dataset"
    ]
    lofi_datasets = [
        "zakarii/lofi-hip-hop-midi",
        "saikayala/jazz-ml-ready-midi",
        "phillipssempeebwa/lo-fi-hip-hop-midis"
    ]
    secondary_genre_datasets = [
        "mmaximssew/metal-midi-dataset"
    ]

    download_kaggle_datasets(
        general_datasets, target_folder="training_data/phase1_general")
    download_kaggle_datasets(
        lofi_datasets, target_folder="training_data/phase2_lofi")
    download_kaggle_datasets(secondary_genre_datasets,
                             target_folder="training_data/phase3_secondary")

    print("\n--- Step 2: Multithreaded Tensor Conversion ---")

    for phase_name, paths in PHASES.items():
        raw_dir = paths["raw_dir"]
        tensor_dir = paths["tensor_dir"]

        if not os.path.exists(raw_dir):
            print(f"Skipping {phase_name}: {raw_dir} does not exist.")
            continue

        os.makedirs(tensor_dir, exist_ok=True)

        raw_files = get_midi_files(raw_dir)
        print(f"\nFound {len(raw_files)} files in {raw_dir}")

        # Skip conversion if we already have the tensors (like Phase 1)
        # Works as a simple file counter
        existing_tensors = get_midi_files(tensor_dir)
        if os.path.exists(tensor_dir) and len(os.listdir(tensor_dir)) > 0:
            print(f"Tensors already exist for {
                  phase_name}. Skipping conversion.")
            continue

        process_args = [(f, tensor_dir) for f in raw_files]

        successful_conversions = 0
        with ProcessPoolExecutor(max_workers=14) as executor:
            results = list(tqdm(executor.map(process_single_file, process_args), total=len(
                raw_files), desc=f"Converting {phase_name}"))

            successful_conversions = sum(1 for r in results if r)

        print(f"Successfully converted {
              successful_conversions} files to tensors in {tensor_dir}")
