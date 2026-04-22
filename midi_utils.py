import os
import sys
import shutil
import warnings
from tqdm import tqdm
import pretty_midi
import kagglehub


def parse_midi_file(file_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                inst_name = pretty_midi.program_to_instrument_name(
                    instrument.program)
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end,
                        'instrument_program': instrument.program,
                        'instrument_name': inst_name
                    })
        return sorted(notes, key=lambda x: x['start'])
    except Exception:
        return None


def download_kaggle_datasets(dataset_identifiers, target_folder="midi_dataset"):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    stored_paths = []
    for dataset_id in dataset_identifiers:
        print(f"Downloading dataset: {dataset_id}...")
        try:
            cache_path = kagglehub.dataset_download(dataset_id)
            dataset_name = dataset_id.split('/')[-1]
            final_path = os.path.join(target_folder, dataset_name)

            if not os.path.exists(final_path):
                shutil.copytree(cache_path, final_path)
                print(f"Successfully stored {dataset_id} in: {final_path}")
            else:
                print(f"Dataset already exists in: {final_path}")

            stored_paths.append(final_path)
        except Exception as e:
            print(f"Failed to download {dataset_id}. Error: {e}")

    return stored_paths


def get_midi_files(directory):
    midi_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, file))
    return midi_files


def filter_valid_midi_files(file_paths, min_notes=50):
    valid_files = []
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, module="pretty_midi")
    print(f"Scanning {len(file_paths)} files for errors...")

    for file_path in tqdm(file_paths):
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            notes = parse_midi_file(file_path)
            if notes is not None and len(notes) >= min_notes + 1:
                valid_files.append(file_path)
        except Exception:
            pass
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

    print(f"\nKept {len(valid_files)} valid files.")
    return valid_files


def sequence_to_midi(sequence, output_filename='generated_track.mid'):
    midi = pretty_midi.PrettyMIDI()
    instruments_dict = {}

    current_time = 0.0
    for note_features in sequence:
        pitch = int(max(0, min(127, note_features[0].item())))
        velocity = int(max(0, min(127, note_features[1].item())))
        duration = note_features[2].item()
        inst_program = int(max(0, min(127, note_features[3].item())))

        # Extract the new Delta Time feature
        delta = max(0.0, note_features[4].item())

        # Move the timeline forward by delta, NOT duration!
        current_time += delta

        if inst_program not in instruments_dict:
            new_inst = pretty_midi.Instrument(program=inst_program)
            instruments_dict[inst_program] = new_inst
            midi.instruments.append(new_inst)

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + duration
        )
        instruments_dict[inst_program].notes.append(note)

    midi.write(output_filename)
    print(f"Saved generated music to {output_filename}")
    return output_filename
