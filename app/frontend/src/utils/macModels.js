// On a Mac the MLX models stand in for the PyTorch fast ones, and Medium's
// PyTorch builds need an NVIDIA GPU. Anything already here stays listed.
export const hiddenOnMac = (c) => !c.downloaded && !c.active_job
    && (c.kind === 'post-trained' || c.hardware === 'cuda+flash-attn');
