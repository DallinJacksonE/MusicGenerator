# genre_config.py

# 1. Base Pre-training Configuration
BASE_CONFIG = {
    "name": "Phase 1 General",
    "kaggle_datasets": [
        "soumikrakshit/classical-music-midi",
        "kritanjalijain/maestropianomidi",
        "joebeachcapital/guitar-chords-midi-pitches",
        "imsparsh/lakh-midi-clean",
        "hansespinosa2/40000-video-game-midi-files",
        "alexignatov/the-expanded-groove-midi-dataset"
    ],
    "raw_dir": "training_data/phase1_general",
    "tensor_dir": "tensor_data/phase1_general",
    "epochs": 30,
    "lr": 1e-4,
    "target_bpm": 120,
    "normalize_key": False,
    "weights_file": "phase1_weights.pth"
}

# 2. The Genre Registry (The "Factory" Data)
GENRE_REGISTRY = {
    "lofi": {
        "kaggle_datasets": [
            "zakarii/lofi-hip-hop-midi",
            "saikayala/jazz-ml-ready-midi",
            "phillipssempeebwa/lo-fi-hip-hop-midis"
        ],
        "raw_dir": "training_data/genre_lofi",
        "tensor_dir": "tensor_data/genre_lofi",
        "epochs": 50,
        "lr": 1e-5,
        "target_bpm": 75,
        "normalize_key": True,
        "allowed_instruments": [0, 4, 5, 32]  # Piano, EPiano, Bass
    },
    "metal": {
        "kaggle_datasets": [
            "mmaximssew/metal-midi-dataset"
        ],
        "raw_dir": "training_data/genre_metal",
        "tensor_dir": "tensor_data/genre_metal",
        "epochs": 15,
        "lr": 1e-5,
        "target_bpm": 140,
        "normalize_key": True,
        # Overdrive Guitar, Distortion Guitar, Picked Bass, Synth Drum
        "allowed_instruments": [29, 30, 34, 118]
    }
    # Want to add jazz? Just drop a "jazz": {...} block right here!
}
