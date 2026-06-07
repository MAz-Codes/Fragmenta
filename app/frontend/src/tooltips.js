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
};
