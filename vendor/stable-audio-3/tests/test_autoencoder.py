import numpy as np
import torch
import torch.nn.functional as F

from tests.utils.audio import assert_audio_valid, sine_wave

DURATION_SEC = 2


def test_encode_decode(autoencoder):
    ae = autoencoder
    channels = ae.autoencoder.io_channels

    audio = sine_wave(
        DURATION_SEC, ae.sample_rate, channels=channels, device=ae.device
    )  # (C, T)

    latents = ae.encode(audio, ae.sample_rate)
    recon = ae.decode(latents)

    assert_audio_valid(recon, DURATION_SEC, ae.sample_rate)

    # Loose reconstruction check: cosine similarity between flattened signals
    min_len = min(audio.shape[-1], recon.shape[-1])
    orig_flat = audio[..., :min_len].flatten().float()
    recon_flat = recon[0, ..., :min_len].flatten().float()
    similarity = F.cosine_similarity(
        orig_flat.unsqueeze(0), recon_flat.unsqueeze(0)
    ).item()
    assert similarity > 0.5, (
        f"Reconstruction cosine similarity too low: {similarity:.3f}"
    )


def test_preprocess_for_encoder(autoencoder):
    """preprocess_audio_for_encoder resamples and pads a single (C, T) clip."""
    ae = autoencoder
    channels = ae.autoencoder.io_channels

    # Generate at half the model rate to exercise resampling
    input_sr = ae.sample_rate // 2
    audio = sine_wave(DURATION_SEC, input_sr, channels=channels)  # (C, T)

    batch = ae.autoencoder.preprocess_audio_for_encoder(audio, in_sr=input_sr)

    assert batch.ndim == 3, f"Expected (1, C, T), got {batch.shape}"
    assert batch.shape[0] == 1
    assert batch.shape[1] == channels
    # Output length should be a multiple of the downsampling ratio
    assert batch.shape[2] % ae.autoencoder.downsampling_ratio == 0, (
        f"Padded length {batch.shape[2]} is not a multiple of "
        f"downsampling_ratio {ae.autoencoder.downsampling_ratio}"
    )


def test_preprocess_list_for_encoder(autoencoder):
    """preprocess_audio_list_for_encoder handles clips of different lengths."""
    ae = autoencoder
    channels = ae.autoencoder.io_channels

    clip_a = sine_wave(1, ae.sample_rate, channels=channels)  # 1 s
    clip_b = sine_wave(DURATION_SEC, ae.sample_rate, channels=channels)  # 2 s

    batch = ae.autoencoder.preprocess_audio_list_for_encoder(
        [clip_a, clip_b], in_sr_list=ae.sample_rate
    )

    assert batch.ndim == 3, f"Expected (2, C, T), got {batch.shape}"
    assert batch.shape[0] == 2
    assert batch.shape[1] == channels
    assert batch.shape[2] % ae.autoencoder.downsampling_ratio == 0


def test_chunked_encode_decode(autoencoder):
    """encode / decode with chunked=True produce valid audio."""
    ae = autoencoder
    channels = ae.autoencoder.io_channels

    audio = sine_wave(
        DURATION_SEC, ae.sample_rate, channels=channels, device=ae.device
    )  # (C, T)

    latents = ae.encode(audio, ae.sample_rate, chunked=True, chunk_size=128, overlap=32)
    recon = ae.decode(latents, chunked=True, chunk_size=128, overlap=32)

    assert_audio_valid(recon, DURATION_SEC, ae.sample_rate)


def test_latent_save_load(autoencoder, tmp_path):
    """Latents saved as .npy and reloaded decode to valid audio."""
    ae = autoencoder
    channels = ae.autoencoder.io_channels

    audio = sine_wave(
        DURATION_SEC, ae.sample_rate, channels=channels, device=ae.device
    )  # (C, T)
    latents = ae.encode(audio, ae.sample_rate)

    # Save and reload via numpy
    latent_path = tmp_path / "latents.npy"
    np.save(latent_path, latents[0].cpu().numpy())

    latent_tensor = torch.from_numpy(np.load(latent_path)).unsqueeze(0).to(ae.device)
    assert latent_tensor.shape == latents.shape

    recon = ae.decode(latent_tensor)
    assert_audio_valid(recon, DURATION_SEC, ae.sample_rate)
