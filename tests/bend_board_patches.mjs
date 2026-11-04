// Emit the exact bend modules the Probe board sends for every contact and
// for a few wires, using the frontend's own code (bendUtils.js), so the
// backend verification (verify_bend_board.py) checks what the board really
// does rather than a hand-written imitation of it.
//
//   node tests/bend_board_patches.mjs registry.json > patches.json
//
// registry.json is the /api/bend/targets payload (targets + operators +
// swap_fns), dumped by verify_bend_board.py from the Python registry.
import { readFileSync } from 'node:fs';
import {
    boardColumns, probeModules, jumperPadId,
} from '../app/frontend/src/components/bend/bendUtils.js';

const registry = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const wet = Number(process.argv[3] || 0.8);     // the board's default dry/wet
const columns = boardColumns(registry);

const out = [];
for (const col of columns) {
    for (const pad of col.pads) {
        out.push({ pads: [pad.id], wet, modules: probeModules([pad.id], columns, registry, wet) });
    }
}
// Wires: a forward block→block wire, a forward wire landing on a part, a
// feedback wire, and a self-loop; depth-aware so they exist on any model.
const dit = registry.stages.find(s => s.stage === 'dit');
const n = dit ? dit.count : 0;
if (n >= 4) {
    const wires = [
        jumperPadId(1, Math.min(n - 1, 9)),                       // forward, block→block
        jumperPadId(2, Math.min(n - 1, 7), 'ff', 'attn'),         // forward, ff→attention
        jumperPadId(Math.min(n - 1, 12), 3),                      // feedback (previous call)
        jumperPadId(5, 5, 'block', 'block'),                      // self loop
        jumperPadId(3, 4),                                        // no-op wire: must be dropped
    ];
    for (const id of wires) {
        out.push({ pads: [id], wet, modules: probeModules([id], columns, registry, wet) });
    }
    // Two contacts held together with a wire, as a chord.
    const chord = ['dit.4', 'latent.1', jumperPadId(1, Math.min(n - 1, 9))];
    out.push({ pads: chord, wet, modules: probeModules(chord, columns, registry, wet) });
}
process.stdout.write(JSON.stringify(out, null, 1));
