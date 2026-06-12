// Normalize backend error payloads into a display string.
//
// The backend speaks two error dialects: plain `{error: "message"}` from the
// simpler routes, and the APIResponse envelope `{error: {message, code,
// details}}` from the validated ones. Rendering the envelope object directly
// as a React child crashes the tree ("Objects are not valid as a React
// child"), so every catch block that surfaces a message to the UI should run
// it through here.
export function extractError(e, fallback = 'Something went wrong') {
    const data = e?.response?.data;
    const err = data?.error;
    if (typeof err === 'string' && err) return err;
    if (err && typeof err === 'object' && typeof err.message === 'string' && err.message) {
        return err.message;
    }
    if (typeof data?.message === 'string' && data.message) return data.message;
    if (typeof data?.detail === 'string' && data.detail) return data.detail;
    if (typeof e?.message === 'string' && e.message) return e.message;
    return fallback;
}
