import json
import os
import logging

logging.basicConfig(level=logging.WARNING)

def _load_metadata():
    metadata = {}
    here = os.path.dirname(__file__)
    candidates = [
        # Current layout: <repo>/vendor/stable-audio-tools/custom_metadata.py
        # → metadata at <repo>/data/metadata.json (two levels up).
        os.path.join(here, '..', '..', 'data', 'metadata.json'),
        # Pre-vendor-move layout: <repo>/stable-audio-tools/custom_metadata.py
        # → metadata at <repo>/data/metadata.json (one level up).
        os.path.join(here, '..', 'data', 'metadata.json'),
        # Legacy: data was under app/backend/.
        os.path.join(here, '..', '..', 'app', 'backend', 'data', 'metadata.json'),
        os.path.join(here, '..', 'app', 'backend', 'data', 'metadata.json'),
    ]

    chosen = next((p for p in candidates if os.path.exists(p)), None)
    if chosen is None:
        logging.warning(f"Metadata file not found in any of: {candidates}")
        return metadata

    try:
        with open(chosen, 'r') as jsonfile:
            metadata_list = json.load(jsonfile)
            for item in metadata_list:
                metadata[item['file_name']] = item['prompt']
    except Exception as e:
        logging.warning(f"Error loading metadata from {chosen}: {e}")

    return metadata

def get_custom_metadata(info, audio):
    metadata = _load_metadata()
    actual_filename = info.get("filename") or os.path.basename(info["relpath"])
    if actual_filename in metadata:
        prompt = metadata[actual_filename]
    else:
        logging.warning(
            f"Filename {actual_filename} not found in metadata JSON. Using empty prompt.")
        prompt = ""
    return {"prompt": prompt}