async function request(method, url, body, config = {}) {
    const init = { method };
    const headers = { ...(config.headers || {}) };

    if (body !== undefined && body !== null) {
        if (body instanceof FormData) {
            init.body = body;
        } else {
            init.body = JSON.stringify(body);
            if (!headers['Content-Type']) headers['Content-Type'] = 'application/json';
        }
    }
    if (Object.keys(headers).length > 0) init.headers = headers;

    const response = await fetch(url, init);

    let data;
    if (config.responseType === 'blob') {
        data = await response.blob();
    } else {
        const text = await response.text();
        try { data = text ? JSON.parse(text) : null; }
        catch { data = text; }
    }

    if (!response.ok) {
        const err = new Error(`HTTP ${response.status}`);
        err.response = { status: response.status, data };
        throw err;
    }
    return { data, status: response.status };
}

const api = {
    get: (url, config) => request('GET', url, null, config),
    post: (url, body, config) => request('POST', url, body, config),
    put: (url, body, config) => request('PUT', url, body, config),
    delete: (url, config) => request('DELETE', url, null, config),
};

export default api;
