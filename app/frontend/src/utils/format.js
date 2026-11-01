export function formatDuration(seconds) {
    const sec = Math.floor(seconds % 60);
    const min = Math.floor((seconds / 60) % 60);
    const hr = Math.floor(seconds / 3600);
    return [hr, min, sec]
        .map((v, i) => (i === 0 ? v : v.toString().padStart(2, '0')))
        .join(':');
}

export function formatTimeLeft(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    if (seconds < 90) return 'under 2 min left';
    if (seconds < 3600) return `~${Math.round(seconds / 60)} min left`;
    return `~${(seconds / 3600).toFixed(1)} h left`;
}

// hh:mm, rounded up so a running estimate never reads 00:00.
export function formatHoursMinutes(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    const minutes = Math.ceil(seconds / 60);
    return `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
}
