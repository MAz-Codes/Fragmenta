"""Fragmenta Bend — network bending & model bending core.

Network bending (inference-time) in the lineage of Broad et al. (2021,
2022: deterministic transformation layers on activations, clustered feature
selection), Kotowski & Font (NIME 2026: realtime bending of weights and
biases, the Bending Log, Model Blending) and Dzwonczyk, Cella & Ban (DAFx
2024: network bending inside a diffusion model's denoiser). The working
method — modify, listen, log, no hypothesis first — is Reed Ghazala's
circuit-bending "anti-theory", which Gillespie & Schachter (DAFx 2022)
carried into modelled analog circuits and Kotowski & Font into neural
synthesisers. Every claim about what these functions do, and what was
measured, is in phd/network-bending.md (local, untracked).

Public surface:
    operators.OPERATORS               operator registry (pure tensor fns)
    patch.validate_patch(...)         schema validation + normalization
    targets.get_targets(model_id)     bendable-point registry for the UI
    session.BendSession               apply/teardown on a loaded model
    lora_bend.bend_lora(...)          adapter-file bending & Model Blending
    bendlog.BendLog                   the Bending Log store
"""
