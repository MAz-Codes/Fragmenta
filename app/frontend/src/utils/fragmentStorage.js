// Audio of every Performance channel fragment. Fragments are too large for
// localStorage (1–3 MB each, up to 200 per channel), so the small metadata
// sits in the session and only the raw Blob is stored here: as files in the
// data folder (see performanceStore.js), or, on the Space, in the window's
// IndexedDB.
//
// Records are stored under a `scope` + `id` pair. Two scope families exist:
//   • `session-ch{N}`        — live session blobs for channel N (0..3).
//   • `preset-{name}-ch{N}`  — frozen snapshot owned by a named preset.
// Save/load/delete preset operations copy or clear by scope; reload reads
// the `session-ch{N}` scope to rehydrate channels.
//
// Before 2.0 the audio was only in IndexedDB. moveFragmentsOver() moves it to
// the data folder once; until it has, reads fall back to IndexedDB.
//
// NOTE: The IndexedDB database name stays `fragmenta-takes` for backwards
// compatibility — renaming would orphan everything users have generated.
// The on-disk key/scope strings don't contain "take" anywhere; only this
// module's exported function names and the database name carry the legacy.
import api from '../api';
import { performanceFilesAvailable } from './performanceStore';

const DB_NAME = 'fragmenta-takes';
const DB_VERSION = 1;
const STORE = 'blobs';

let dbPromise = null;
// Set once the window's audio has moved to the data folder (or there was none).
let legacyGone = false;

function openDB() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
        // Opening would create the deleted database again, empty.
        if (legacyGone && performanceFilesAvailable()) {
            dbPromise = null;
            reject(new Error('Fragments have moved to the data folder'));
            return;
        }
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

// Whether the window still has audio from before 2.0, without creating the
// database just to look.
async function hasLegacyDB() {
    if (legacyGone || typeof indexedDB === 'undefined') return false;
    if (typeof indexedDB.databases !== 'function') return true;
    const names = (await indexedDB.databases()).map((d) => d.name);
    if (!names.includes(DB_NAME)) legacyGone = true;
    return !legacyGone;
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

function idbPut(scope, id, blob) {
    return withStore('readwrite', (store) => new Promise((resolve, reject) => {
        const req = store.put({ key: compoundKey(scope, id), scope, id, blob });
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
    }));
}

function idbGet(scope, id) {
    return withStore('readonly', (store) => new Promise((resolve, reject) => {
        const req = store.get(compoundKey(scope, id));
        req.onsuccess = () => resolve(req.result?.blob || null);
        req.onerror = () => reject(req.error);
    }));
}

function idbDelete(scope, id) {
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

async function idbClearScope(scope) {
    const items = await getAllByScope(scope);
    if (items.length === 0) return;
    return withStore('readwrite', (store) => {
        items.forEach(it => store.delete(it.key));
    });
}

// Copy every blob from one scope to another, preserving the per-fragment id.
// Existing records at the destination scope are not cleared first — call
// clearScope(toScope) beforehand if a replace is desired.
async function idbCopyScope(fromScope, toScope) {
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

const scopeUrl = (scope) => `/api/performance/fragments/${encodeURIComponent(scope)}`;
const fragmentUrl = (scope, id) => `${scopeUrl(scope)}/${encodeURIComponent(id)}`;

async function filePut(scope, id, blob) {
    const r = await fetch(fragmentUrl(scope, id), {
        method: 'PUT',
        headers: { 'Content-Type': blob.type || 'application/octet-stream' },
        body: blob,
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
}

export async function putFragmentBlob(scope, id, blob) {
    if (!performanceFilesAvailable()) return idbPut(scope, id, blob);
    return filePut(scope, id, blob);
}

export async function getFragmentBlob(scope, id) {
    if (!performanceFilesAvailable()) return idbGet(scope, id);
    const r = await fetch(fragmentUrl(scope, id));
    if (r.ok) return r.blob();
    if (r.status !== 404) throw new Error(`HTTP ${r.status}`);
    // Not moved over yet: still in the window.
    const blob = (await hasLegacyDB()) ? await idbGet(scope, id).catch(() => null) : null;
    if (blob) {
        filePut(scope, id, blob).catch(() => { /* moveFragmentsOver retries */ });
        return blob;
    }
    // The move may have finished meanwhile: look in the folder once more.
    const again = await fetch(fragmentUrl(scope, id));
    return again.ok ? again.blob() : null;
}

export async function deleteFragmentBlob(scope, id) {
    if (!performanceFilesAvailable()) return idbDelete(scope, id);
    await api.delete(fragmentUrl(scope, id));
    // So a copy not moved over yet can't come back.
    if (await hasLegacyDB()) await idbDelete(scope, id).catch(() => {});
}

export async function clearScope(scope) {
    if (!performanceFilesAvailable()) return idbClearScope(scope);
    await api.delete(scopeUrl(scope));
    if (await hasLegacyDB()) await idbClearScope(scope).catch(() => {});
}

// Move one scope's audio from the window to the data folder, skipping what's
// already there.
async function moveScopeOver(scope, have) {
    if (!(await hasLegacyDB())) return;
    for (const it of await getAllByScope(scope)) {
        if (!have.has(compoundKey(it.scope, it.id))) await filePut(it.scope, it.id, it.blob);
    }
}

async function filesHave() {
    const { data } = await api.get('/api/performance/fragments');
    return new Set((data?.fragments || []).map((f) => compoundKey(f.scope, f.id)));
}

export async function copyScope(fromScope, toScope) {
    if (!performanceFilesAvailable()) return idbCopyScope(fromScope, toScope);
    // A preset saved before 2.0 may still have its audio in the window.
    if (await hasLegacyDB()) await moveScopeOver(fromScope, await filesHave());
    await api.post(`${scopeUrl(fromScope)}/copy`, { to: toScope });
}

/** Once: move all the window's fragment audio into the data folder, then
 *  delete it from the window. Safe to run again after a failure. */
export async function moveFragmentsOver() {
    if (!performanceFilesAvailable() || !(await hasLegacyDB())) return;
    const have = await filesHave();
    const keys = await withStore('readonly', (store) => new Promise((resolve, reject) => {
        const req = store.getAllKeys();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => reject(req.error);
    }));
    // One record at a time: a channel can hold hundreds of MB.
    for (const key of keys) {
        if (have.has(key)) continue;
        const it = await withStore('readonly', (store) => new Promise((resolve, reject) => {
            const req = store.get(key);
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        }));
        if (it?.blob) await filePut(it.scope, it.id, it.blob);
    }
    const db = await openDB();
    db.close();
    dbPromise = null;
    legacyGone = true;
    await new Promise((resolve) => {
        const req = indexedDB.deleteDatabase(DB_NAME);
        req.onsuccess = req.onerror = req.onblocked = () => resolve();
    });
}

export const channelScope = (channelIndex) => `session-ch${channelIndex}`;
export const presetChannelScope = (presetName, channelIndex) =>
    `preset-${encodeURIComponent(presetName)}-ch${channelIndex}`;
