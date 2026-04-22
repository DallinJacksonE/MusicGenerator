import os
import torch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from midi_utils import get_midi_files, parse_midi_file, download_kaggle_datasets
from genre_config import BASE_CONFIG, GENRE_REGISTRY
SEQ_LENGTH = 128


def process_single_file(args):
    """Parses a single MIDI file and saves it as a PyTorch tensor."""
    file_path, tensor_dir, target_bpm, normalize_key = args

    try:
        notes = parse_midi_file(file_path, target_bpm, normalize_key)

        if not notes or len(notes) < SEQ_LENGTH + 1:
            return False

        sequence = []
        current_time = notes[0]['start'] if notes else 0.0

        for note in notes[:SEQ_LENGTH + 1]:
            duration = note['end'] - note['start']
            delta = note['start'] - current_time
            current_time = note['start']

            # We now append 5 features instead of 4
            sequence.append([note['pitch'], note['velocity'],
                            duration, note['instrument_program'], delta])

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

    # Download Base Phase 1
    download_kaggle_datasets(
        BASE_CONFIG["kaggle_datasets"], target_folder=BASE_CONFIG["raw_dir"])

    # Dynamically download all registered genres
    for genre, config in GENRE_REGISTRY.items():
        download_kaggle_datasets(
            config["kaggle_datasets"], target_folder=config["raw_dir"])

    print("\n--- Step 2: Multithreaded Tensor Conversion ---")

    # Combine Base Config and Genre Configs into one list for processing
    all_phases = [BASE_CONFIG] + list(GENRE_REGISTRY.values())

    for phase in all_phases:
        raw_dir = phase["raw_dir"]
        tensor_dir = phase["tensor_dir"]
        phase_name = phase.get("name", f"Genre: {raw_dir.split('_')[-1]}")

        if not os.path.exists(raw_dir):
            print(f"Skipping {phase_name}: {raw_dir} does not exist.")
            continue

        os.makedirs(tensor_dir, exist_ok=True)
        raw_files = get_midi_files(raw_dir)
        print(f"\nFound {len(raw_files)} files in {raw_dir}")

        if os.path.exists(tensor_dir) and len(os.listdir(tensor_dir)) > 0:
            print(f"Tensors already exist for {
                  phase_name}. Skipping conversion.")
            continue

        target_bpm = phase.get("target_bpm", None)
        normalize_key = phase.get("normalize_key", False)

        # Append it to the tuple passed to the thread pool
        process_args = [(f, tensor_dir, target_bpm, normalize_key)
                        for f in raw_files]
        with ProcessPoolExecutor(max_workers=14) as executor:
            results = list(tqdm(executor.map(process_single_file,
                                             process_args),
                                total=len(raw_files),
                                desc=f"Converting {phase_name}"))
            successful_conversions = sum(1 for r in results if r)

        print(f"Successfully converted {
              successful_conversions} files to tensors in {tensor_dir}")
