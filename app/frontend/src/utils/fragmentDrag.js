// In-app drag handoff for generated fragments.
//
// The HTML drag-and-drop dataTransfer can only carry strings, so when a
// fragment is dragged from the Generated Fragments window into the Edit tab's
// source dropzone we stash its in-memory audio Blob here on dragStart and read
// it back on drop. This lets EditPanel use the blob directly instead of
// re-fetching by filename — which is immune to any divergence between the
// fragment's in-memory name and what actually exists on disk.
//
// Falls back gracefully: if no blob was stashed (e.g. a not-yet-preloaded
// disk fragment), the consumer drops back to the filename-based fetch.

let _payload = null; // { filename: string, blob: Blob } | null

export function setFragmentDragPayload(payload) {
    _payload = payload;
}

export function getFragmentDragPayload() {
    return _payload;
}

export function clearFragmentDragPayload() {
    _payload = null;
}
