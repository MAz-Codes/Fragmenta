export function formatDuration(seconds) {
    const sec = Math.floor(seconds % 60);
    const min = Math.floor((seconds / 60) % 60);
    const hr = Math.floor(seconds / 3600);
    return [hr, min, sec]
        .map((v, i) => (i === 0 ? v : v.toString().padStart(2, '0')))
        .join(':');
}
