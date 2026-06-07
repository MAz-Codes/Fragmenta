// IndexedDB store for per-channel fragment audio blobs. Fragments are too
// large for localStorage (1–3 MB each, up to 200 per channel) so we keep
// the small metadata in the localStorage session and only the raw Blob here.
//
// Records are stored under a `scope` + `id` pair. Two scope families exist:
//   • `session-ch{N}`        — live session blobs for channel N (0..3).
//   • `preset-{name}-ch{N}`  — frozen snapshot owned by a named preset.
// Save/load/delete preset operations copy or clear by scope; reload reads
// the `session-ch{N}` scope to rehydrate channels.
//
// NOTE: The IndexedDB database name stays `fragmenta-takes` for backwards
// compatibility — renaming would orphan everything users have generated.
// The on-disk key/scope strings don't contain "take" anywhere; only this
// module's exported function names and the database name carry the legacy.

const DB_NAME = 'fragmenta-takes';
const DB_VERSION = 1;
const STORE = 'blobs';

let dbPromise = null;

function openDB() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
        if (typeof indexedDB === 'undefined') {
            reject(new Error('IndexedDB not available'));
            return;
        }
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE)) {
                const store = db.createObjectStore(STORE, { keyPath: 'key' });
                store.createIndex('scope', 'scope', { unique: false });
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
    return dbPromise;
}

const compoundKey = (scope, id) => `${scope}::${id}`;

function withStore(mode, fn) {
    return openDB().then(db => new Promise((resolve, reject) => {
        const txn = db.transaction(STORE, mode);
        const store = txn.objectStore(STORE);
        let result;
        txn.oncomplete = () => resolve(result);
        txn.onerror = () => reject(txn.error);
        txn.onabort = () => reject(txn.error);
        Promise.resolve(fn(store)).then((r) => { result = r; }, reject);
    }));
}

export async function putFragmentBlob(scope, id, blob) {
    return withStore('readwrite', (store) => new Promise((resolve, reject) => {
        const req = store.put({ key: compoundKey(scope, id), scope, id, blob });
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    }));
}

export async function getFragmentBlob(scope, id) {
    return withStore('readonly', (store) => new Promise((resolve, reject) => {
        const req = store.get(compoundKey(scope, id));
        req.onsuccess = () => resolve(req.result?.blob || null);
        req.onerror = () => reject(req.error);
    }));
}

export async function deleteFragmentBlob(scope, id) {
    return withStore('readwrite', (store) => new Promise((resolve, reject) => {
        const req = store.delete(compoundKey(scope, id));
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    }));
}

// Read every record whose scope field matches exactly. Used at preset copy
// time and for full-scope clears.
async function getAllByScope(scope) {
    return withStore('readonly', (store) => new Promise((resolve, reject) => {
        const idx = store.index('scope');
        const req = idx.getAll(IDBKeyRange.only(scope));
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => reject(req.error);
    }));
}

export async function clearScope(scope) {
    const items = await getAllByScope(scope);
    if (items.length === 0) return;
    return withStore('readwrite', (store) => {
        items.forEach(it => store.delete(it.key));
    });
}

// Copy every blob from one scope to another, preserving the per-fragment id.
// Existing records at the destination scope are not cleared first — call
// clearScope(toScope) beforehand if a replace is desired.
export async function copyScope(fromScope, toScope) {
    const items = await getAllByScope(fromScope);
    if (items.length === 0) return;
    return withStore('readwrite', (store) => {
        items.forEach(it => store.put({
            key: compoundKey(toScope, it.id),
            scope: toScope,
            id: it.id,
            blob: it.blob,
        }));
    });
}

export const channelScope = (channelIndex) => `session-ch${channelIndex}`;
export const presetChannelScope = (presetName, channelIndex) =>
    `preset-${encodeURIComponent(presetName)}-ch${channelIndex}`;
