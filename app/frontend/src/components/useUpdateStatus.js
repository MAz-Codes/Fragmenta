import { useCallback, useEffect, useState } from 'react';
import api from '../api';

// The backend checks GitHub once a day and caches the answer
// (app/backend/update_check.py); this hook only reads that cache, so polling
// is cheap and never touches the network itself.
const POLL_MS = 6 * 60 * 60 * 1000;
// The backend's first check runs ~30 s after it starts — pick that up.
const FIRST_RESULT_MS = 45 * 1000;

export default function useUpdateStatus() {
    const [status, setStatus] = useState(null);
    const [checking, setChecking] = useState(false);

    const refresh = useCallback(async () => {
        try {
            const r = await api.get('/api/update/status');
            setStatus(r.data);
        } catch {
            // Backend still starting — the next poll will catch up.
        }
    }, []);

    useEffect(() => {
        refresh();
        const first = setTimeout(refresh, FIRST_RESULT_MS);
        const poll = setInterval(refresh, POLL_MS);
        return () => { clearTimeout(first); clearInterval(poll); };
    }, [refresh]);

    const post = useCallback(async (path, body) => {
        const r = await api.post(path, body || {});
        setStatus(r.data);
        return r.data;
    }, []);

    const checkNow = useCallback(async () => {
        setChecking(true);
        try { await post('/api/update/check'); } catch { /* shown via last_error */ } finally { setChecking(false); }
    }, [post]);

    const skip = useCallback(() => post('/api/update/dismiss', { action: 'skip' }), [post]);
    const snooze = useCallback(() => post('/api/update/dismiss', { action: 'snooze' }), [post]);
    const setSettings = useCallback((settings) => post('/api/update/settings', settings), [post]);
    // The backend picks the URL from its validated state; we only name the button.
    const open = useCallback((target) => api.post('/api/update/open', { target }).catch(() => {}), []);

    const visible = Boolean(status && status.available && !status.skipped
        && !status.snoozed && !status.disabled_reason);

    return { status, visible, checking, refresh, checkNow, skip, snooze, setSettings, open };
}
