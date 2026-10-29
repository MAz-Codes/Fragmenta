# AI transparency

Fragmenta generates audio with AI models. Under the EU AI Act (Article 50(2)), audio made by an AI system must be marked in a machine-readable way and be detectable as AI-generated. This page explains what Fragmenta does, how to check a file, and where the limits are.

## How outputs are marked

Since version 1.1.1, every generated WAV and every saved performance recording carries metadata tags (WAV `LIST/INFO`): a title, the software, and a comment such as

```
AI-generated with Fragmenta 1.1.1; model=sa3-small-music-base; adapters=[my_lora/epoch=21-step=500];
digital_source_type=http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia
```

The comment names the model and any LoRA adapters, and ends with the IPTC digital source type, a public vocabulary that machines can read: `trainedAlgorithmicMedia` for generations, `compositeWithTrainedAlgorithmicMedia` for performance recordings, which are live mixes of generated fragments. The audio itself is not changed.

## How to check a file

Any tool that shows WAV tags will do, free and offline — for example:

```bash
ffprobe -hide_banner your_file.wav     # FFmpeg; look for "comment"
exiftool your_file.wav                 # ExifTool
```

MediaInfo and many DAWs and tag editors show the same comment.

## Limits

Metadata can be lost: converting a file, exporting it again from a DAW, or uploading it to some platforms may drop the tags. Please keep them when you share audio made with Fragmenta.

## Why there is no audio watermark

In October 2026 we tested Meta's AudioSeal watermark on Fragmenta's own generations. At a strength that survives MP3 conversion, it was audible in a blind listening test on music; it sits only about 25–32 dB below the music and above it in sparse frequency bands. Weaker settings did not survive MP3, and shaping it to stay under the music made it undetectable. For a music tool, where the sound is the product, an audible watermark is not a workable solution, so Fragmenta does not add one. We will revisit this as watermarking for music improves.

## What comes next

Fragmenta 2.0 adds **signed, time-stamped metadata** (C2PA Content Credentials) to exported files, verifiable with public C2PA tools. Like the tags above, it doesn't touch the audio.

## Code of Practice

Fragmenta has not signed the EU Code of Practice on Transparency of AI-Generated Content. Compared with its measures for providers:

| Code measure | Fragmenta |
|---|---|
| At least two machine-readable marking techniques | Metadata now; signed metadata (C2PA) in 2.0; no watermark (see above) |
| Signed, time-stamped metadata | 2.0 (C2PA) |
| A free way to detect the marks | Standard tag readers now; public C2PA verifiers from 2.0 |
| Interoperable detection | Public formats: WAV INFO tags and the IPTC vocabulary now, the C2PA standard in 2.0 |
| Users asked not to remove the marks | This page and the README now; the end-user terms in 2.0 |

Questions: please open an issue on the [GitHub repository](https://github.com/MAz-Codes/Fragmenta).
