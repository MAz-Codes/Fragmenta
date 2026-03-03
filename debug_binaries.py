
from PyInstaller.utils.hooks import collect_all

packages = ['scipy', 'torchaudio', 'librosa']

for pkg in packages:
    print(f"--- Collecting {pkg} ---")
    try:
        datas, hiddenimports, binaries = collect_all(pkg)
        
        print(f"Binaries Count: {len(binaries)}")
        if binaries:
            print(f"Sample Binary: {binaries[0]}")
            
        # Check for weird binaries
        for i, b in enumerate(binaries):
            if not isinstance(b, tuple) or len(b) != 2:
                print(f"!!! FOUND ABNORMAL BINARY at index {i}: {b} (Type: {type(b)}, Len: {len(b)})")
                break
        else:
            print("All binaries are 2-element tuples.")
            
        # Check weird hidden imports in torchaudio again
        if pkg == 'torchaudio':
             for item in hiddenimports:
                 if not isinstance(item, str):
                     print(f"Torchaudio Hidden Import Item: {item} (Len: {len(item)})")

    except Exception as e:
        print(f"Error collecting {pkg}: {e}")
