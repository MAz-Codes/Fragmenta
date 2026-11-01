export function formatDuration(seconds) {
    const sec = Math.floor(seconds % 60);
    const min = Math.floor((seconds / 60) % 60);
    const hr = Math.floor(seconds / 3600);
    return [hr, min, sec]
        .map((v, i) => (i === 0 ? v : v.toString().padStart(2, '0')))
        .join(':');
}

export function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    if (seconds < 90) return 'under 2 min';
    if (seconds < 3600) return `~${Math.round(seconds / 60)} min`;
    return `~${(seconds / 3600).toFixed(1)} h`;
}

export function formatTimeLeft(seconds) {
    const time = formatTime(seconds);
    return time && `${time} left`;
}
