import json
import os
import logging

logging.basicConfig(level=logging.WARNING)

def _load_metadata():
    metadata = {}
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, '..', 'data', 'metadata.json'),               # current layout
        os.path.join(here, '..', 'app', 'backend', 'data', 'metadata.json'),  # legacy
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