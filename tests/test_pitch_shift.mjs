// Runs tests/test_pitch_shift.html in headless Chromium and prints the checks.
//
// Run:  node tests/test_pitch_shift.mjs [path-to-chromium]
import { spawn } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const CHROME = process.argv[2] || ['/snap/bin/chromium', '/usr/bin/chromium', '/usr/bin/google-chrome']
    .find((p) => fs.existsSync(p));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript' };

const server = http.createServer((req, res) => {
    const file = path.join(REPO, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (!file.startsWith(REPO) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404); res.end(); return;
    }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/tests/test_pitch_shift.html`;

// Snap Chromium can only write under ~/snap.
const profile = fs.mkdtempSync(path.join(os.homedir(), CHROME.startsWith('/snap') ? 'snap/chromium/common/pitch-' : '.pitch-test-'));
const port = 9300 + Math.floor(Math.random() * 500);
// Own process group: the snap wrapper forks the real browser, so killing just
// the wrapper would leave Chromium writing into the profile we delete.
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, url],
    { stdio: 'ignore', detached: true });

let fails = 0;
try {
    let page;
    for (let i = 0; i < 50 && !page; i++) {
        await new Promise((r) => setTimeout(r, 200));
        try { page = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === 'page'); } catch { /* not up yet */ }
    }
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((r) => ws.addEventListener('open', r, { once: true }));
    let id = 0;
    const pending = new Map();
    ws.addEventListener('message', (ev) => {
        const m = JSON.parse(ev.data);
        if (pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
    });
    const evaluate = (expression) => new Promise((resolve) => {
        const i = ++id;
        pending.set(i, (r) => resolve(r?.result?.value));
        ws.send(JSON.stringify({ id: i, method: 'Runtime.evaluate', params: { expression, returnByValue: true } }));
    });
    let results = null;
    for (let i = 0; i < 300 && !results; i++) {
        await new Promise((r) => setTimeout(r, 200));
        if (await evaluate('document.title') === 'DONE') results = await evaluate('window.__results');
    }
    ws.close();
    if (!results) { console.log('  [FAIL] page never finished'); fails++; }
    for (const r of results || []) {
        console.log(`  [${r.ok ? 'PASS' : 'FAIL'}] ${r.name}${r.detail ? ' — ' + r.detail : ''}`);
        if (!r.ok) fails++;
    }
} finally {
    try { process.kill(-chrome.pid, 'SIGTERM'); } catch { /* already gone */ }
    await new Promise((r) => setTimeout(r, 500));
    server.close();
    fs.rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}
console.log(fails ? `\n${fails} FAILED` : '\nAll pitch-shift checks passed.');
process.exit(fails ? 1 : 0);
