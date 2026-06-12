async function request(method, url, body, config = {}) {
    const init = { method };
    const headers = { ...(config.headers || {}) };

    if (body !== undefined && body !== null) {
        if (body instanceof FormData) {
            init.body = body;
            // Let the browser set Content-Type so the multipart boundary is included.
            delete headers['Content-Type'];
        } else {
            init.body = JSON.stringify(body);
            if (!headers['Content-Type']) headers['Content-Type'] = 'application/json';
        }
    }
    if (Object.keys(headers).length > 0) init.headers = headers;
    // Forward the caller's AbortController signal so in-flight requests
    // (e.g. a running generation) are actually cancellable.
    if (config.signal) init.signal = config.signal;

    const response = await fetch(url, init);

    let data;
    if (config.responseType === 'blob' && response.ok) {
        data = await response.blob();
    } else {
        // Error bodies are always JSON/text even on blob requests — reading
        // them as a blob would hide the backend's message from the user.
        const text = await response.text();
        try { data = text ? JSON.parse(text) : null; }
        catch { data = text; }
    }

    // Fetch lowercases header names, so consumers can read
    // responseHeaders['x-fragment-filename'] etc. directly.
    const responseHeaders = {};
    response.headers.forEach((value, key) => { responseHeaders[key] = value; });

    if (!response.ok) {
        const err = new Error(`HTTP ${response.status}`);
        err.response = { status: response.status, data, headers: responseHeaders };
        throw err;
    }
    return { data, status: response.status, headers: responseHeaders };
}

const api = {
    get: (url, config) => request('GET', url, null, config),
    post: (url, body, config) => request('POST', url, body, config),
    put: (url, body, config) => request('PUT', url, body, config),
    patch: (url, body, config) => request('PATCH', url, body, config),
    delete: (url, config) => request('DELETE', url, null, config),
};

export default api;
