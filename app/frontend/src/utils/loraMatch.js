// SA3 LoRA <-> base-model compatibility check.
//
// A LoRA trained against e.g. `sa3-small-music-base` should be usable when
// generating with either `sa3-small-music-base` (training-style runs) or
// `sa3-small-music` (the post-trained / distilled sibling). They share the
// same backbone — only the CFG state differs. Strict `===` would treat
// them as different and hide trained LoRAs from the generation picker.
//
// Strips a trailing `-base` from both IDs before comparing, so:
//   sa3-small-music        ↔ sa3-small-music-base       → compatible
//   sa3-medium             ↔ sa3-medium-base            → compatible
//   sa3-small-music        ↔ sa3-medium                 → not compatible
//   sa3-small-music        ↔ sa3-small-sfx              → not compatible
function baseRoot(modelId) {
    if (!modelId) return '';
    return modelId.endsWith('-base') ? modelId.slice(0, -5) : modelId;
}

export function isLoraCompatible(loraBase, generationModel) {
    if (!loraBase || !generationModel) return false;
    return baseRoot(loraBase) === baseRoot(generationModel);
}

export function filterLorasForModel(loras, generationModel) {
    if (!Array.isArray(loras) || !generationModel) return [];
    return loras.filter(l => isLoraCompatible(l.base_model, generationModel));
}
