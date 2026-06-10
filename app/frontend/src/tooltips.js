export const TIPS = {
    // App.js — LoRA training hyperparameters + model row actions.
    training: {
        downloadModel: 'Download this model',
        deleteFineTuned: 'Delete fine-tuned model',
        steps: "SA3's documented quick-start is 1,000 steps.",
        adapter: "DoRA-rows is SA3's upstream default and works best for most stylistic LoRAs. The -xs variants freeze SVD bases and only train a tiny core matrix — far fewer parameters, useful when VRAM is tight. BoRA scales both rows and columns independently (more expressive, more parameters).",
        checkpointEvery: 'How often a LoRA .safetensors snapshot gets written. Auto picks ~10 checkpoints per run (capped 250–1 000 steps). Lower = more granular but more disk; higher = fewer files to compare.',
        batchSize: 'SA3 examples use 1. Each extra sample adds ~1–2 GB of activations. Raise only on roomy GPUs (≥24 GB); medium-base activations are heavy. Lower if you hit CUDA OOM.',
        precision: 'Cast applied to the frozen base weights only; LoRA parameters stay in fp32 for the optimizer. bf16 halves the VRAM used by the base with negligible quality cost on Ampere and newer cards.',
        rank: "Capacity of the LoRA update — rank-k matrices A (k×in) and B (out×k) are trained. Higher rank = more expressive but larger file and more VRAM. r=16 fits comfortably on 16 GB and is SA3's default.",
        alpha: 'Scaling factor for the LoRA update. Effective scaling is alpha / rank — setting alpha = rank gives a scaling of 1.0. Conventional choice: alpha = rank.',
        dropout: 'Regularization probability applied to LoRA inputs during training. 0 is fine for most cases — raise to ~0.05 if you see overfitting on small datasets.',
        seed: 'Random seed for reproducibility — same dataset + same hyperparameters + same seed produces the same LoRA. Change it to re-roll with different sampling behaviour.',
        learningRate: "AdamW step size for the LoRA weights (base stays frozen). SA3's default is 1e-4, which works for most runs. Too high destabilizes training (loss spikes, artifacts); too low barely moves the adapter. Halve it if loss is erratic.",
        sampleLength: 'Audio fed to the model per training step. Long clips get random-cropped to this length each step; short clips get silence-padded. Capped at the base model\'s native length (~120s small, ~380s medium) — longer windows cost markedly more VRAM and step time, so raise it only for long-form material (pre-encoding helps).',
        includeLayers: 'Space-separated substrings — only layers whose fully-qualified name contains one of these get LoRA. Empty = all matching Linear/Conv1d layers. Example: transformer.layers.',
        excludeLayers: 'Space-separated substrings — matching layers are skipped, even if they also match Include. SA3-docs default (seconds_total to_local_embed) prevents conditioner-hijacking on small datasets.',
    },

    // PerformancePanel.js — top transport bar + bottom controls.
    perf: {
        notDownloaded: 'Not downloaded — open Checkpoint Manager',
        midiSettings: 'MIDI settings & mappings',
        presets: 'Save / load presets',
        deletePreset: 'Delete preset',
        launchQuant: "Launch quantization — match Ableton's",
        deleteFineTuned: 'Delete fine-tuned model',
        deleteLora: 'Delete LoRA',
        promptKey: 'Auto-inject Key. Leave empty to skip.',
        timeSig: 'Auto-inject Time signature. Leave empty to skip.',
        link: ({ installing, available, enabled, peers }) =>
            installing
                ? 'Installing LinkPython-extern…'
                : !available
                    ? 'Click to install Ableton Link script'
                    : enabled
                        ? `Link on — ${peers} peer${peers === 1 ? '' : 's'} (click to disable)`
                        : 'Click to sync BPM with Ableton Link',
        midiMode: ({ supported, permissionError, learnMode }) =>
            !supported
                ? (permissionError || 'Web MIDI is not available')
                : learnMode
                    ? 'Exit MIDI mode (Esc)'
                    : 'Enter MIDI mode — click a control then move a hardware knob/button to bind',
        audioSetup: (cueSupported) =>
            cueSupported
                ? 'Audio setup — choose output device'
                : 'Audio device selection requires Chrome/Edge (AudioContext.setSinkId). Output falls back to system default.',
        restoreDefaults: (armed) =>
            armed
                ? 'Click again within 3s to confirm — clears session, fragments, and MIDI mappings'
                : 'Reset all panel settings, clear fragments, and clear MIDI mappings',
        steps: (isDistilled) =>
            isDistilled
                ? 'Locked at 8 steps for distilled SA3 models — pick a *-base checkpoint to override'
                : 'Diffusion steps per generation (more = higher quality, slower)',
        bpmInject: (on, bpm) =>
            on
                ? `Injecting master BPM (${Math.round(bpm)}) into prompts — click to disable`
                : 'Click to auto-inject the master BPM (top bar) into every prompt',
    },

    // PerformanceChannel.js — per-channel strip.
    channel: {
        mute: 'Mute',
        solo: 'Solo',
        batch: "Batch generate Fragments and cue below.",
        loop: (looping, durationMode) =>
            looping
                ? (durationMode === 'bars'
                    ? 'Loop'
                    : 'Playback loop on')
                : 'Loop off',
        generateDisabled: (generating, canGenerate, hasPrompt) =>
            generating
                ? ''
                : !canGenerate
                    ? 'Pick a model in the Generation tab first'
                    : !hasPrompt
                        ? 'Enter a prompt to generate'
                        : '',
        variation: (loaded) =>
            loaded
                ? 'Variation from the current fragment'
                : 'Generate a fragment first, then create variations of it',
    },

    // DatasetPrep.js — dataset workbench.
    dataset: {
        autoAnnotateAll: 'Generate a prompt for every clip in the project at once, using the current annotation tier and template. Existing prompts are kept or overwritten per the "Skip annotated" switch.',
        templatePreset: 'The sentence shape used when building prompts from detected tags. Music is full tracks, Instrument/Stem is single parts, Sample/SFX is one-shots — each follows SA3\'s tag convention for that material.',
        richAnnotate: 'Adds genre / mood / instrument tags using LAION-CLAP. Requires the CLAP weights — downloadable from the Checkpoint Manager.',
        skipAnnotated: 'When on, Auto-annotate skips clips that already have an annotation. Off means every run overwrites existing prompts.',
        deleteProject: 'Delete this project (folder, audio, sidecars, drafts) — irreversible',
        discardChanges: 'Delete unsaved changes — reverts to the last created dataset (removes any audio added since)',
        saveDraft: "Save a draft — persists across app restarts but isn't the SA3 sidecar form",
        createDataset: 'Create Dataset — writes the .txt sidecars (overwrites the previous dataset)',
        selectClips: 'Click to select these clips — then Auto-annotate them.',
        autoAnnotateClip: 'Auto-annotate this clip (overwrites any current prompt)',
        sliceClip: 'Slice this clip into shorter children (immediate)',
        removeClip: 'Remove this clip from the project (immediate)',
        tooShort: (thresholdSec) =>
            `Shorter than ${thresholdSec}s — gets silence-padded into each batch. Consider deleting. Click to select.`,
        duplicates: (count) =>
            `${count} group${count === 1 ? '' : 's'} of clips share the same annotation. Bad for training diversity — click to select all of them.`,
        unsupported: (accepted) =>
            `SA3 only trains on ${(accepted || []).join(', ')}. These clips will be silently skipped at train time — re-export them as .wav (or another accepted format) before committing. Click to select.`,
    },

    // LoraStack.js — LoRA slot stack.
    lora: {
        stackInfo: (max) => `Blend up to ${max} LoRAs at any strength`,
        dragReorder: 'Drag to reorder (slot 0 loads first)',
        bypass: (bypassed) =>
            bypassed ? 'Bypassed (strength 0) — click to enable' : 'Bypass this slot',
    },

    // Fragment lists — ChannelFragmentHistory.js + GeneratedFragmentsWindow.js.
    fragments: {
        clearAll: 'Clear all (delete every fragment from disk)',
        deleteFromDisk: 'Delete from disk',
        revealInFolder: 'Show in folder (reveal this file on disk)',
        download: 'Download to your computer',
        audition: (isAuditioning) =>
            isAuditioning ? 'Stop cue' : 'Audition through cue output',
        star: (starred) =>
            starred ? 'Unstar' : 'Star (keep through eviction)',
        commit: (committed) =>
            committed ? 'Currently loaded' : 'Load into channel',
    },

    // CheckpointRow.js — checkpoint catalog rows.
    checkpoints: {
        gatedAccess: "Open on HuggingFace to accept the model's gated-access terms",
    },

    // App.js — main Audio Generation tab (the primary "create" surface).
    generate: {
        mode: 'Generate new makes audio from a text prompt. Edit existing transforms a clip you provide (style transfer, inpaint a region, or extend it).',
        modelSelect: 'The checkpoint that does the generating. Distilled models are fixed at fast settings (8 steps, no guidance); -base models let you tune CFG and steps. Greyed-out rows aren\'t downloaded yet — use the download icon.',
        prompt: 'Describe the sound you want — instruments, genre, mood, tempo, key. SA3 responds to AudioSparx-style tags (e.g. "TrackType: Music, Genre: Techno, BPM: 128") as well as plain English.',
        duration: 'Length of the clip to generate, in seconds. Capped at the model\'s native maximum (~120s for small models, ~380s for medium). Longer clips take proportionally longer to render.',
        negativePrompt: 'Optional. Describe what to steer away from (e.g. "vocals, distortion, silence"). Only used on -base models that support guidance; ignored by distilled models.',
        cfg: 'Classifier-Free Guidance — how strictly the model follows your prompt. Low (1–3) is loose and creative; high (8–15) hugs the prompt but can sound harsh. SA3\'s sweet spot is ~7. Only on -base models.',
        steps: 'How many denoising steps the sampler runs. More steps = cleaner detail but slower; fewer = faster but rougher. 50 is a good default for -base models. Distilled models are locked at 8.',
        batch: 'How many clips to generate from this one prompt in a single run. Each uses its own seed (unless you fix the seed). Handy for auditioning variations.',
        seed: 'The random starting point. Random rolls a fresh seed every clip (the value is recorded on each fragment). Turn it off and set a number to reproduce an exact result, or to vary one setting at a time.',
        generateButton: 'Render the clip(s) with the current settings. Disabled until you\'ve picked a downloaded model and entered a prompt.',
        stop: 'Stop the current generation. The in-progress clip is discarded — nothing is saved.',
    },

    // EditPanel.js — "Edit existing" mode (audio-to-audio).
    edit: {
        source: 'The clip to transform. Drop a file here or click to browse (wav/mp3/flac/m4a/ogg/opus). Everything below works on this audio.',
        mode: 'Style transfer re-imagines the whole clip toward your prompt. Inpaint replaces just a selected region, keeping the rest. Extend continues the clip past its end.',
        initNoise: 'How far to push the source toward the prompt. Low keeps the original\'s character and timing; high lets the prompt dominate (closer to generating from scratch).',
        maskRegion: 'The slice that gets regenerated — drag on the waveform or type start/end seconds. Everything outside it is preserved, and the output stays the same total length.',
        extendSeconds: 'How many seconds of new audio to append after the source. The model continues from the existing material toward your prompt.',
        prompt: 'Describes the target sound for this edit. For Inpaint it only steers the selected region; for Style/Extend it steers the whole result.',
        generateButton: 'Run the edit on the source clip with the settings above. Disabled until a source clip is loaded.',
    },

    // MidiConfigMenu.js — MIDI device + mapping config.
    midi: {
        device: 'Which connected MIDI controller Fragmenta listens to. "None" disables MIDI input. Unplugged-but-remembered devices reconnect automatically when plugged back in.',
        channelFilter: 'Which MIDI channel to accept. "All" reacts to every channel; pick a specific one (1–16) when several devices share the port or you want to isolate a controller.',
        takeover: 'How a knob takes control when its hardware position differs from the on-screen value. Jump snaps to the hardware on first move (can leap). Pickup waits until you sweep through the current value, so there\'s no jump.',
        mappings: 'Every hardware control you\'ve bound to a Fragmenta control. Enter MIDI mode, click a control, then move a knob/button to add one here.',
        clearAll: 'Remove every MIDI mapping at once. The device connection and filters stay.',
        removeMapping: 'Delete just this one mapping.',
    },

    // CheckpointManagerWindow.js — model download manager.
    manager: {
        storage: 'See how much disk each downloaded model and checkpoint is using, and free space by removing ones you don\'t need.',
        hfLogin: 'Sign in to HuggingFace so Fragmenta can fetch gated or private checkpoints on your account. Stored locally; used only for downloads.',
        hfLogout: 'Forget the saved HuggingFace token on this machine.',
        hfToken: 'Paste a HuggingFace access token (starts with "hf_"). Create one at huggingface.co/settings/tokens — read scope is enough for downloads.',
        refresh: 'Re-scan the catalog and disk for which models are downloaded, in case something changed outside the app.',
    },

    // TrainingMonitor.js — training status card.
    monitor: {
        lossChart: 'Training loss per step, lower is better. A curve that falls then flattens means the LoRA is converging; a flat or rising curve suggests the learning rate or data needs attention.',
        steps: 'Progress through the run, counted in optimizer steps (SA3 trains by steps, not epochs).',
        checkpoints: 'How many LoRA snapshots have been written so far. Each is a usable adapter you can generate with or keep training from.',
    },
};
