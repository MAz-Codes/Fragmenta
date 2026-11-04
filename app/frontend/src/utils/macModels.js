// The one PyTorch model the Bend tab bends (verified contact by contact in
// tests/verify_bend_board.py). Bending hooks PyTorch modules, so MLX can't
// be bent; on a Mac this model is offered next to its MLX twin, for the
// Bend tab only, and runs on the PyTorch/MPS path there.
export const BEND_MODEL = 'sa3-small-music';
export const bendOnlyOnMac = (c) => c.id === BEND_MODEL;

// On a Mac the MLX models stand in for the PyTorch fast ones, and Medium's
// PyTorch builds need an NVIDIA GPU. Anything already here stays listed,
// and so does the Bend tab's model.
export const hiddenOnMac = (c) => !c.downloaded && !c.active_job && !bendOnlyOnMac(c)
    && (c.kind === 'post-trained' || c.hardware === 'cuda+flash-attn');
