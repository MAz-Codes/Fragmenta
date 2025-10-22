import json
import os
import logging

logging.basicConfig(level=logging.WARNING)

def _load_metadata():
    metadata = {}
    json_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'backend', 'data', 'metadata.json')
    
    try:
        with open(json_path, 'r') as jsonfile:
            metadata_list = json.load(jsonfile)
            for item in metadata_list:
                metadata[item['file_name']] = item['prompt']
    except FileNotFoundError:
        logging.warning(f"Metadata file not found: {json_path}")
    except Exception as e:
        logging.warning(f"Error loading metadata: {e}")
    
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