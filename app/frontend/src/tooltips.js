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
        sampleLength: 'Audio fed to the model per training step. Clips LONGER than this get a fresh random crop each step; clips shorter get padded and train on the same excerpt every time. Keep it under your shortest clips — that random cropping is the main augmentation a small dataset has. Long windows also cost markedly more VRAM and step time, so raise it only for genuinely long-form material.',
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
        linkSync: "Link sync. Use Offset only for delay Fragmenta can't measure, e.g. Bluetooth headphones.",
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
        bendPreset: 'Generate this channel through a bend preset saved in the Bend tab. The bend applies to that generation only.',
        bendAmount: 'Bend amount — scales every module of the preset from clean (left) to full (right). Structural bends switch on past halfway. MIDI-mappable.',
        mute: 'Mute',
        solo: 'Solo',
        sidechain: (active, locked) =>
            locked
                ? 'Another channel is the sidechain source'
                : active
                    ? 'Sidechain on — this channel ducks all the others'
                    : 'Sidechain — duck all other channels under this one',
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
        closeProject: 'Close this project — nothing is deleted. Reopen it anytime via Load project.',
        discardChanges: 'Delete unsaved changes — reverts to the last created dataset (removes any audio added since)',
        saveDraft: "Save a draft — persists across app restarts but isn't the SA3 sidecar form",
        createDataset: 'Create Dataset — writes the .txt sidecars (overwrites the previous dataset)',
        injectText: 'Text to splice into many annotations at once — a shared detail like "recorded at 120bpm" or a mic note. Press Enter or hit Inject. Repeat as often as you like; each pass keeps what the last one wrote.',
        injectMode: 'Where the injected text lands. Append puts it after the existing annotation, Prepend before it, Replace discards the existing annotation entirely. Empty annotations just get the bare text, with no stray comma.',
        injectScope: 'Which clips get the text — every clip in the project, or only the ones ticked in the table. Switches to Selected automatically as soon as you tick something.',
        injectUndo: 'Roll back the last injection, one pass at a time. Clips you\'ve edited by hand since are left as they are, so undo never eats newer work.',
        selectClips: 'Click to select these clips — then Auto-annotate them.',
        emptyPrompts: 'Clips without a prompt are left out of pre-encoding and training. Click to select them, then Auto-annotate or write prompts.',
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
        rename: 'Double-click to rename (empty resets to F#)',
        moveUp: 'Move up',
        moveDown: 'Move down',
    },

    // CheckpointRow.js — checkpoint catalog rows.
    checkpoints: {
        gatedAccess: "Open on HuggingFace to accept the model's gated-access terms",
        openOnHf: 'Open the model page on HuggingFace',
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
        extendSeconds: 'How many seconds of new audio to append after the source. The model continues from the existing material toward your prompt. If the source fades out at its end, the new audio replaces that fade.',
        prompt: 'Describes the target sound for this edit. For Inpaint it only steers the selected region, and the surrounding audio matters more than the prompt: a small region stays close to the original, so select more for a bigger change. For Style it steers the whole result; for Extend, the new part.',
        generateButton: 'Run the edit on the source clip with the settings above. Disabled until a source clip is loaded.',
    },

    // MidiConfigMenu.js — MIDI device + mapping config.
    midi: {
        device: 'The MIDI controllers Fragmenta listens to. Switch on as many as you like; each mapping remembers the controller it was learned from. A switched-on controller reconnects when plugged back in.',
        channelFilter: 'Which MIDI channel to take from this controller. "Any" takes every channel; pick one (1–16) to ignore the rest.',
        takeover: 'How a knob takes control when its hardware position differs from the on-screen value. Jump snaps to the hardware on first move (can leap). Pickup waits until you sweep through the current value, so there\'s no jump.',
        mappings: 'Every hardware control you\'ve bound to a Fragmenta control. Enter MIDI mode, click a control, then move a knob/button to add one here.',
        clearAll: 'Remove every MIDI mapping at once. The device connection and filters stay.',
        removeMapping: 'Delete just this one mapping.',
    },

    // CheckpointManagerWindow.js — model download manager.
    manager: {
        storage: 'See how much disk each downloaded model and checkpoint is using, and free space by removing ones you don\'t need.',
        hfLogin: 'Sign in to HuggingFace so Fragmenta can fetch gated or private checkpoints on your account. Stored locally; used only for downloads.',
        hfLoginDemo: 'Disabled in this demonstration Space. Install Fragmenta locally from misaghazimi.com/fragmenta to sign in with your own HuggingFace account.',
        hfLogout: 'Forget the saved HuggingFace token on this machine.',
        hfToken: 'Paste a HuggingFace access token (starts with "hf_"). Create one at huggingface.co/settings/tokens — read scope is enough for downloads.',
    },

    // TrainingMonitor.js — training status card.
    monitor: {
        lossChart: 'Training loss per step, lower is better. A curve that falls then flattens means the LoRA is converging; a flat or rising curve suggests the learning rate or data needs attention.',
        steps: 'Progress through the run, counted in optimizer steps (SA3 trains by steps, not epochs).',
        timeLeft: 'Hours and minutes left, estimated from the speed over the last two minutes.',
        checkpoints: 'How many LoRA snapshots have been written so far. Each is a usable adapter you can generate with or keep training from.',
    },

    // Bend tab — network bending (Bend) + model bending (Break). In the
    // lineage of circuit bending: intervene in the model itself and listen.
    bend: {
        modeBend: 'Bend: intervene inside the frozen model while it generates — its activations, weights, latents and structure. Reversible, per-generation.',
        modeBreak: 'Break: train adapters wrong on purpose — underfit, overfit, corrupted lessons. Produces bent LoRA files.',
        model: 'The model being bent. Only Small - Music (the fast PyTorch build, 8 steps) is offered: it is the one every contact has been checked on, and it keeps the touch→listen loop tight. MLX models cannot be bent.',
        chance: "Ghazala's anti-theory protocol as a button: randomize the rack and listen, no hypothesis required. Nudge is subtle, Snap is not.",
        unbend: 'Clear the rack. (The model itself is never left bent — every bend is applied for exactly one generation and fully removed.)',
        presets: 'Save the current rack as a named patch, or load one. A patch + seed reproduces a bent sound exactly — a bend saved is an instrument kept.',
        generate: 'Generate with the rack applied. The bend exists only for this generation; the fragment and its patch land in the Bending Log below.',
        abCompare: 'Regenerate the same seed with the rack bypassed — bending is only legible against the un-bent baseline.',
        duration: 'Short clips keep the modify→listen loop immediate. Stretch it out once a bend is worth living inside.',
        signalPath: 'The model, drawn as a signal path. Click a stage to attach a bend module there: the prompt\u2019s meaning, the step embedding, the 20 transformer blocks, the sound-in-progress between steps, and the decoder nearest the loudspeaker.',
        log: 'The Bending Log: every bent generation is recorded with its full patch and seed. Add a one-line note of what it did to the sound — the log becomes your instrument vocabulary, and any row recalls its bend.',
        viewProbe: 'Board: the model as a board of unlabelled contacts. Touch one and listen — no theory needed, nothing to choose from a menu.',
        viewRack: 'Rack: the case opened. Every bend as a module with its target, operator and parameters, for when you want to choose the what as well as the where.',
        showWiring: 'Reveal what each contact is connected to inside the model. Off by default — find out by ear first.',
        dryWet: 'How much of the bent signal gets through at every contact and wire — dry is the untouched sound, wet is fully bent. Moving it re-renders the board as it is wired and held.',
        newInput: 'Pick a new seed: the board gets new material to bend. Contacts behave the same; the sound passing through them changes.',
        freshInput: 'Random seed: every press starts from a new seed, so the same contact never sounds the same twice. Each press also renders its own clean version for A/B, so it takes about twice as long. Off: every press uses the same seed, which makes contacts easy to compare.',
        logAll: 'Show the log for every model, not just the one on the board. A row from another model recalls onto this board: the same contacts, that model\u2019s sound if it is offered, otherwise this one\u2019s.',
        clearBoard: 'Release every held contact and remove every wire. Your named finds stay — touch one to bring it back.',
        keep: 'Mark this find. Its name goes on the contact you touched and into the Bending Log.',
        openInRack: 'Open the case: load this sound into the Rack, with every module and parameter visible and editable.',
        live: 'Hear the bend as you change it. Off: touches and Generate make takes, as usual. Loop: one clip (this seed, prompt and length) loops, and every change re-renders it — the same material, bent differently. Stream: the model keeps continuing what was just played, with the bend as it is now; the sound evolves instead of looping. Nothing is saved or logged while this is on.',
        liveChunk: 'How much the model continues at a time. Shorter chunks follow the knobs sooner; longer ones give it more to work with.',
    },
    // components/engine/EnginePanel.js — the perpetual engine as a tab.
    engine: {
        start: 'Start the engine as its own process (the same thing `python -m app.engine` runs). The model loads in the background; the tab talks to the engine\u2019s own API on port 5002.',
        stopEngine: 'Stop the engine process. Its state is saved every few seconds, so the next start resumes the scene and settings.',
        play: 'Start perpetual playback: beds render ahead and play, events are scheduled by the conductor.',
        stopTransport: 'Stop playback. The model stays loaded.',
        record: 'Record the master output to a WAV in the data folder (engine/recordings), signed like every other export.',
        listen: 'Hear the engine\u2019s output in this window, streamed from the engine. A few seconds behind the real output.',
        model: 'A base checkpoint: blends and guidance only act on base models (the distilled ones ignore CFG). The SFX base is the likely prior for field recordings.',
        lora: 'Adapters trained for this base model (others are hidden). None = the plain base model, which already lets you hear whether a blend moves the sound.',
        checkpoint: 'Which saved step of the run to load. Earlier steps are safer against memorisation, later ones follow the data more closely; pick by ear.',
        layout: 'Where the sound goes. Stereo now; binaural and speaker layouts decode to stereo until the HOA decoder exists.',
        loadModel: 'Load this model and LoRA into the running engine. Anchors are re-encoded for the new checkpoint.',
        transition: 'How long the next scene takes to arrive: the old bed recedes while the new one enters.',
        fromAzimuth: 'Where the next scene enters from. 0 = front, positive = left, like walking from the bazaar into a side street.',
        blend: 'Drag between the anchors. Every bed take shares this blend and differs only by seed; events are rendered close, from the same blend.',
        wander: 'Let the blend drift by itself through anchor space.',
        oneshot: 'Render ten seconds from the current blend and play it here, to hear what a blend sounds like before committing to it.',
        device: 'The audio interface the engine plays through (PortAudio). None keeps the engine silent except for Listen and the radio.',
        channels: 'How many output channels to open on the device. Routes beyond this number are dropped.',
        route: 'Where this layer goes: Spatial = through the decoder (stereo now, speaker layouts later); an arrow = straight to those output channels, for mixing or processing outside.',
        radioUrl: 'Where the stream goes: an Icecast mount (icecast://source:password@host:8000/tehran.mp3). The codec follows the extension: .mp3 or .opus. The engine reconnects by itself if the link drops.',
        clock: 'The scene follows Tehran\u2019s time of day from the radio scene file, so a restart lands in the right place. Off: scenes are chosen here or by the sequence.',
        program: 'The piece as a list of scenes: which place, for how many minutes, how it arrives, and whether the beds are generated, played from the real-recording pool, or both. Saved as a file in the data folder, so another dataset makes another piece. The blend pad moves between places by hand.',
        programLoad: 'Make this list the one playing. It starts at its first scene; the loop runs until you stop it.',
        programSave: 'Save the list as shown. A list that is playing picks up the edit at once.',
        programNext: 'Skip to the next scene now, with its own transition.',
        programLoop: 'At the end, start again from the first scene. Off: the last scene stays until you change it.',
        programBeds: 'Where the beds of this scene come from. Generated = the model; Pool = real recordings and pre-rendered beds from the pool folder; Both = mixed. The pool player arrives with E4.6.',
        knobs: {
            '/events/density': 'How often events happen. The most important knob after Silence.',
            '/events/sparseness': 'The chance that a scheduled event is skipped: silence as a parameter.',
            '/events/max_concurrent': 'Event budget: how many events may sound at once.',
            '/events/min_interval': 'Minimum time between event onsets, with jitter.',
            '/bed/takes': 'Decorrelated takes of the bed playing together: same blend, different seeds.',
            '/bed/clip_length': 'Length of each rendered bed clip. Longer = fewer renders, more GPU time per render.',
            '/bed/crossfade_s': 'Crossfade between consecutive bed clips.',
            '/spatial/events/distance_max': 'How far away an event may be placed.',
            '/render/lora_strength': 'Strength of the LoRA for every following render.',
            '/master/gain': 'Master level, before the limiter.',
        },
    },
};
