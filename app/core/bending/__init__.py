"""Fragmenta Bend — network bending & model bending core.

Network bending (inference-time) in the lineage of Broad et al. (activation
bending), Kotowski & Font (parameter-level bending), and Gillespie &
Schachter (model bending / anti-theory). See BEND_PLAN.md at the repo root.

Public surface:
    operators.OPERATORS               operator registry (pure tensor fns)
    patch.validate_patch(...)         schema validation + normalization
    targets.get_targets(model_id)     bendable-point registry for the UI
    session.BendSession               apply/teardown on a loaded model
    lora_bend.bend_lora(...)          adapter-file bending & Model Blending
    bendlog.BendLog                   the Bending Log store
"""
