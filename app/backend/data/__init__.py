import warnings

# Clips libsndfile can't read (m4a / AAC) fall back to librosa's audioread
# decoder. The decode works, but librosa warns twice per file (once that
# soundfile failed, once that the fallback is deprecated), which floods the
# log on every waveform, slice and annotation of such a project. Muted
# process-wide here rather than with catch_warnings around each call: the
# backend is threaded and catch_warnings swaps the global filter list.
warnings.filterwarnings("ignore", message="PySoundFile failed", category=UserWarning)
warnings.filterwarnings("ignore", message=r"librosa\.core\.audio\.__audioread_load",
                        category=FutureWarning)
