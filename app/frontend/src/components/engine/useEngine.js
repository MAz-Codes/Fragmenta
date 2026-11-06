import { useCallback, useEffect, useRef, useState } from 'react';
import api from '../../api';

/**
 * The Engine tab's only link to the world: the backend's three process
 * routes (/api/engine/*) and the engine's own /v1 API on its port. Nothing
 * here is private to the UI — every call is one a script could make.
 */
const STATUS_POLL_MS = 2000;
const PROCESS_POLL_MS = 3000;

export async function v1(base, method, path, body) {
    const init = { method, headers: {} };
    if (body !== undefined) {
        init.body = JSON.stringify(body);
        init.headers['Content-Type'] = 'application/json';
    }
    const r = await fetch(`${base}${path}`, init);
    const text = await r.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!r.ok) {
        const err = new Error((data && data.error) || `HTTP ${r.status}`);
        err.status = r.status;
        throw err;
    }
    return data;
}

export default function useEngine({ active }) {
    const [proc, setProc] = useState(null);        // /api/engine/status
    const [status, setStatus] = useState(null);    // /v1/status
    const [meters, setMeters] = useState(null);    // from the stream
    const [liveEvents, setLiveEvents] = useState([]);
    const [log, setLog] = useState([]);            // warnings + notable events
    const [connected, setConnected] = useState(false);
    const esRef = useRef(null);
    // /api/engine/status reports the API url (…/v1); calls below add their own /v1 path.
    const base = proc && proc.running && proc.url ? proc.url.replace(/\/v1\/?$/, '') : null;

    const refreshProc = useCallback(async () => {
        try {
            const r = await api.get('/api/engine/status');
            setProc(r.data);
            return r.data;
        } catch {
            setProc((p) => ({ ...(p || {}), available: false, running: false }));
            return null;
        }
    }, []);

    const refreshStatus = useCallback(async () => {
        if (!base) { setStatus(null); return null; }
        try {
            const s = await v1(base, 'GET', '/v1/status');
            setStatus(s);
            setConnected(true);
            return s;
        } catch {
            setConnected(false);
            return null;
        }
    }, [base]);

    // process polling
    useEffect(() => {
        if (!active) return undefined;
        refreshProc();
        const t = window.setInterval(refreshProc, PROCESS_POLL_MS);
        return () => window.clearInterval(t);
    }, [active, refreshProc]);

    // engine status polling
    useEffect(() => {
        if (!active || !base) { setStatus(null); setConnected(false); return undefined; }
        refreshStatus();
        const t = window.setInterval(refreshStatus, STATUS_POLL_MS);
        return () => window.clearInterval(t);
    }, [active, base, refreshStatus]);

    // the event stream
    useEffect(() => {
        if (!active || !base) {
            if (esRef.current) { esRef.current.close(); esRef.current = null; }
            return undefined;
        }
        const es = new EventSource(`${base}/v1/stream`);
        esRef.current = es;
        const push = (line) => setLog((l) => [...l.slice(-40), { t: Date.now(), text: line }]);
        es.addEventListener('meters', (e) => { try { setMeters(JSON.parse(e.data)); } catch { /* noop */ } });
        es.addEventListener('event', (e) => {
            try {
                const d = JSON.parse(e.data);
                setLiveEvents((evs) => [...evs.filter((x) => x.clip_id !== d.clip_id || x.t !== d.t), { ...d, started: Date.now() }].slice(-24));
            } catch { /* noop */ }
        });
        es.addEventListener('event_end', (e) => {
            try {
                const d = JSON.parse(e.data);
                setLiveEvents((evs) => evs.filter((x) => x.clip_id !== d.clip_id));
            } catch { /* noop */ }
        });
        es.addEventListener('scene', (e) => { try { const d = JSON.parse(e.data); push(`scene → ${d.id} (${d.transition_s}s from ${d.from_azimuth}°)`); } catch { /* noop */ } });
        es.addEventListener('warning', (e) => { try { const d = JSON.parse(e.data); push(`warning: ${d.error || JSON.stringify(d)}`); } catch { /* noop */ } });
        es.addEventListener('model', (e) => { try { const d = JSON.parse(e.data); push(`model ${d.model_id}: ${d.state}${d.error ? ' — ' + d.error : ''}`); refreshStatus(); } catch { /* noop */ } });
        es.addEventListener('render', (e) => { try { const d = JSON.parse(e.data); if (d.state === 'error') push(`render failed: ${d.error}`); } catch { /* noop */ } });
        es.onerror = () => setConnected(false);
        es.onopen = () => setConnected(true);
        return () => { es.close(); if (esRef.current === es) esRef.current = null; };
    }, [active, base, refreshStatus]);

    // drop finished events the stream didn't close (safety)
    useEffect(() => {
        const t = window.setInterval(() => {
            const now = Date.now();
            setLiveEvents((evs) => evs.filter((e) => now - e.started < (e.duration || 8) * 1000 + 2000));
        }, 1000);
        return () => window.clearInterval(t);
    }, []);

    const call = useCallback((method, path, body) => {
        if (!base) return Promise.reject(new Error('engine not running'));
        return v1(base, method, path, body);
    }, [base]);

    const start = useCallback(async (opts) => {
        const r = await api.post('/api/engine/start', opts || {});
        setProc(r.data);
        return r.data;
    }, []);

    const stop = useCallback(async () => {
        const r = await api.post('/api/engine/stop');
        setProc(r.data);
        setStatus(null);
        setMeters(null);
        setLiveEvents([]);
        return r.data;
    }, []);

    return { proc, base, status, meters, liveEvents, log, connected, call, start, stop, refreshProc, refreshStatus };
}
