/**
 * Bend tab shared helpers — module factory, the Chance randomizer
 * (Ghazala's anti-theory protocol as a function), and patch assembly.
 *
 * The module shape mirrors the backend's patch schema exactly
 * (app/core/bending/patch.py); the backend re-validates everything, so
 * these helpers only need to produce well-formed candidates.
 */

let _uid = 0;
export const nextId = () => `b${Date.now().toString(36)}${(_uid++).toString(36)}`;

export const STAGE_ORDER = ['cond', 'timestep', 'dit', 'latent', 'decoder'];

/** Default params for an operator from its backend spec. */
export function defaultParams(opSpec) {
    const out = {};
    for (const [name, desc] of Object.entries(opSpec?.params || {})) {
        if (desc.default !== undefined) out[name] = desc.default;
    }
    return out;
}

/** A fresh module for a stage, with a sensible operator preselected. */
export function newModule(stage, registry, domain = null) {
    const stageInfo = (registry?.stages || []).find(s => s.stage === stage);
    const dom = domain || (stage === 'latent' ? 'latent' : 'activation');
    const ops = (registry?.operators || []).filter(o =>
        o.domains.includes(dom === 'latent' ? 'latent' : dom));
    const op = ops.find(o => o.name === 'scale') || ops[0];
    const mod = {
        id: nextId(),
        enabled: true,
        target: { stage, domain: dom },
        operator: op?.name || 'scale',
        params: defaultParams(op),
        mix: 1.0,
    };
    if (stageInfo?.kind === 'blocks') {
        // Default to the back half of the DiT (texture) / all decoder blocks.
        const n = stageInfo.count || 4;
        mod.target.blocks = stage === 'dit'
            ? Array.from({ length: Math.ceil(n / 4) }, (_, i) => n - 1 - i).reverse()
            : null;
    }
    if (stage === 'dit' || stage === 'latent') {
        mod.steps = { from: 0.0, to: 1.0 };
    }
    if (dom === 'weight') {
        mod.target.param = 'weight';
        if (stage === 'dit') mod.target.weight_slot = 'ff';
    }
    if (dom === 'structure') {
        delete mod.operator;
        delete mod.params;
        delete mod.mix;
        mod.structure = { type: 'bypass' };
    }
    return mod;
}

const CHANCE_LEVELS = {
    nudge: { modules: [1, 1], mixRange: [0.3, 0.6], paramSpread: 0.35, weightChance: 0.15, structureChance: 0.0 },
    bend:  { modules: [1, 2], mixRange: [0.5, 1.0], paramSpread: 0.7,  weightChance: 0.3,  structureChance: 0.15 },
    snap:  { modules: [2, 3], mixRange: [0.8, 1.0], paramSpread: 1.0,  weightChance: 0.4,  structureChance: 0.3 },
};

const rand = (lo, hi, rng = Math.random) => lo + rng() * (hi - lo);
const pick = (arr, rng = Math.random) => arr[Math.floor(rng() * arr.length)];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function randomParams(opSpec, spread, rng = Math.random) {
    const out = {};
    for (const [name, desc] of Object.entries(opSpec?.params || {})) {
        if (desc.type === 'bool') {
            out[name] = rng() < 0.3;
        } else if (desc.type === 'enum') {
            out[name] = rng() < 0.5 ? desc.default : pick(desc.options || [desc.default], rng);
        } else if (desc.type === 'curve') {
            out[name] = Array.from({ length: 6 }, () => rand(0, 2, rng));
        } else if (desc.min !== undefined && desc.max !== undefined) {
            // Blend between the default and a fully random value by spread.
            const rnd = rand(desc.min, desc.max, rng);
            const base = desc.default ?? (desc.min + desc.max) / 2;
            let v = base + (rnd - base) * spread;
            if (desc.type === 'int') v = Math.round(v);
            out[name] = Math.min(desc.max, Math.max(desc.min, v));
        }
    }
    return out;
}

/**
 * The Chance button: modify and listen, no hypothesis required.
 * Returns a fresh list of modules at the given intensity. With `stage`,
 * returns exactly one module rolled for that stage (the per-card re-roll) —
 * domain, blocks and step range are only valid for the stage they were
 * rolled against, so a module can't be re-rolled on one stage and moved.
 */
export function chancePatch(registry, level = 'bend', { stage: onlyStage = null } = {}) {
    const cfg = CHANCE_LEVELS[level] || CHANCE_LEVELS.bend;
    const allStages = registry?.stages || [];
    const stages = onlyStage
        ? allStages.filter(s => s.stage === onlyStage)
        : allStages.filter(s => s.stage !== 'timestep' || Math.random() < 0.3);
    if (!stages.length) return [];
    const count = onlyStage ? 1 : Math.round(rand(cfg.modules[0], cfg.modules[1]));
    const modules = [];
    for (let i = 0; i < count; i++) {
        const stageInfo = pick(stages);
        const stage = stageInfo.stage;
        let domain = stage === 'latent' ? 'latent' : 'activation';
        if (stage !== 'latent' && Math.random() < cfg.weightChance) domain = 'weight';
        if (stage === 'dit' && Math.random() < cfg.structureChance) domain = 'structure';

        const mod = newModule(stage, registry, domain);
        if (domain === 'structure') {
            const type = pick(['bypass', 'repeat', 'swap_nonlinearity', 'jumper']);
            mod.structure = { type };
            if (type === 'repeat') mod.structure.times = pick([2, 3]);
            if (type === 'swap_nonlinearity') mod.structure.fn = pick(registry?.swap_fns || ['sin']);
            const n = stageInfo.count || 8;
            const b = Math.floor(rand(0, n));
            mod.target.blocks = [b, Math.min(n - 1, b + 1)];
            if (type === 'jumper') {
                Object.assign(mod, jumperModule(b, Math.floor(rand(0, n))), { id: mod.id });
                mod.mix = rand(cfg.mixRange[0], cfg.mixRange[1]);
            }
        } else {
            const ops = (registry?.operators || []).filter(o =>
                o.domains.includes(domain === 'latent' ? 'latent' : domain));
            const op = pick(ops);
            mod.operator = op.name;
            mod.params = randomParams(op, cfg.paramSpread);
            mod.mix = rand(cfg.mixRange[0], cfg.mixRange[1]);
            if (stageInfo.kind === 'blocks') {
                const n = stageInfo.count || 4;
                const span = Math.max(1, Math.round(n * rand(0.15, 0.5)));
                const start = Math.floor(rand(0, n - span + 1));
                mod.target.blocks = Array.from({ length: span }, (_, j) => start + j);
            }
            if (Math.random() < 0.4) {
                mod.target.features = { mode: 'random', fraction: rand(0.2, 0.8), seed: Math.floor(Math.random() * 9999) };
            }
            if (mod.steps && Math.random() < 0.4) {
                const a = rand(0, 0.7);
                mod.steps = { from: a, to: Math.min(1, a + rand(0.3, 1)) };
            }
        }
        modules.push(mod);
    }
    return modules;
}

/** Assemble the request patch from rack state. Generation sends only the
 *  enabled modules; presets keep disabled ones (`keepDisabled`) so a saved
 *  rack comes back exactly as it was, switches included. */
export function buildPatch(modules, modelId, seed, name = '', { keepDisabled = false } = {}) {
    return {
        version: 1,
        name,
        model_id: modelId || '',
        seed: Number.isFinite(seed) ? seed : 0,
        modules: keepDisabled ? modules : modules.filter(m => m.enabled !== false),
    };
}

export const STAGE_SHORT = {
    cond: 'Cond', timestep: 'Time', dit: 'DiT', latent: 'Latent', decoder: 'VAE',
};

/** One-line human summary of a module, for chips and the log. */
export function describeModule(mod, registry) {
    const stage = STAGE_SHORT[mod.target?.stage] || mod.target?.stage;
    if (mod.structure?.type === 'jumper') {
        const end = (b, part) => (part && part !== 'block' ? `${b} ${part}` : `${b}`);
        return `${stage} · wire ${end(mod.structure.from, mod.structure.from_part)}→${end(mod.structure.to, mod.structure.to_part)}`;
    }
    if (mod.target?.domain === 'structure') {
        const t = mod.structure?.type || '?';
        return `${stage} · ${t}${mod.structure?.fn ? `(${mod.structure.fn})` : ''}`;
    }
    const op = (registry?.operators || []).find(o => o.name === mod.operator);
    const blocks = mod.target?.blocks;
    const blockStr = Array.isArray(blocks) && blocks.length
        ? ` ${blocks.length > 3 ? `${blocks[0]}–${blocks[blocks.length - 1]}` : blocks.join(',')}`
        : '';
    const domainStr = mod.target?.domain === 'weight' ? ' ·W' : '';
    return `${stage}${blockStr}${domainStr} · ${op?.label || mod.operator}`;
}


// --- Nearby: move the wire a little -----------------------------------------

/**
 * A small step away from a bend that was just heard: same operators, same
 * kind of target, but blocks shifted by one, numeric params jittered by
 * `amount` of their range, mix and step window nudged. Repeating it is a
 * random walk around a sound — the probing gesture, not a fresh dice roll.
 */
export function mutatePatch(modules, registry, amount = 0.15) {
    const stages = registry?.stages || [];
    const ops = registry?.operators || [];
    return modules.map((m) => {
        const next = JSON.parse(JSON.stringify(m));
        const n = stages.find(s => s.stage === next.target?.stage)?.count;
        const step = () => (Math.random() < 0.5 ? -1 : 1);
        if (n && Array.isArray(next.target?.blocks) && Math.random() < 0.5) {
            const d = step();
            const shifted = next.target.blocks.map(b => b + d);
            if (shifted.every(b => b >= 0 && b < n)) next.target.blocks = shifted;
        }
        if (n && next.structure?.type === 'jumper' && Math.random() < 0.5) {
            const end = Math.random() < 0.5 ? 'from' : 'to';
            next.structure[end] = clamp(next.structure[end] + step(), 0, n - 1);
        }
        const spec = ops.find(o => o.name === next.operator);
        for (const [name, desc] of Object.entries(spec?.params || {})) {
            if (desc.min === undefined || desc.max === undefined) continue;
            if (typeof next.params?.[name] !== 'number') continue;
            let v = next.params[name] + rand(-1, 1) * amount * (desc.max - desc.min);
            if (desc.type === 'int') v = Math.round(v);
            next.params[name] = clamp(v, desc.min, desc.max);
        }
        if (typeof next.mix === 'number') next.mix = clamp(next.mix + rand(-0.1, 0.1), 0.05, 1);
        const f = next.target?.features;
        if (f?.mode === 'random') f.fraction = clamp(f.fraction + rand(-0.1, 0.1), 0.05, 1);
        if (next.steps && (next.steps.from > 0 || next.steps.to < 1)) {
            const d = rand(-0.1, 0.1);
            const width = next.steps.to - next.steps.from;
            const from = clamp(next.steps.from + d, 0, 1 - width);
            next.steps = { from, to: from + width };
        }
        return next;
    });
}

// --- Probe board -------------------------------------------------------------
//
// The model as a circuit board of unlabelled contacts. What each contact
// does is fixed — a pure function of the contact — so touching the same
// point always does the same thing. Nothing is chosen from a menu; it is
// found by touching and listening (Ghazala's anti-theory; BEND_PLAN.md).

/** mulberry32 — tiny seeded PRNG, enough for a fixed per-contact behaviour. */
export function seededRandom(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const hashString = (str) => {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
};

// A wire can land on any of a DiT column's three contacts: the whole block
// (top), its feed-forward (middle) or its attention (bottom). Each pad's
// row tells the wire which part it is soldered to.
const PART_SUFFIX = { block: '', ff: 'ff', attn: 'at' };
const SUFFIX_PART = { '': 'block', ff: 'ff', at: 'attn' };
export const PART_LABEL = { block: 'block', ff: 'feed-forward', attn: 'attention' };

/** The pad a wire end sits on. */
export const wireEndPad = (block, part = 'block') =>
    `dit.${block}${part === 'block' ? '' : `.${PART_SUFFIX[part]}`}`;

/** Pad id of a jumper wire, e.g. j.3-12 or j.3ff-12at. */
export const jumperPadId = (from, to, fromPart = 'block', toPart = 'block') =>
    `j.${from}${PART_SUFFIX[fromPart]}-${to}${PART_SUFFIX[toPart]}`;
export const parseJumperId = (id) => {
    const m = /^j\.(\d+)(ff|at)?-(\d+)(ff|at)?$/.exec(id || '');
    return m ? {
        from: Number(m[1]), fromPart: SUFFIX_PART[m[2] || ''],
        to: Number(m[3]), toPart: SUFFIX_PART[m[4] || ''],
    } : null;
};

/** A wire from a block's output to the next block's input joins two
 *  points that are already one — it would change nothing. */
export const isNoopWire = (from, fromPart, to, toPart) =>
    (fromPart || 'block') === 'block' && (toPart || 'block') === 'block' && to === from + 1;

/** Does the wire run with the signal (forward) or back against it
 *  (feedback, carrying the previous step)? Inside a block the signal goes
 *  in → attention → feed-forward → out. */
export function wireIsForward(from, fromPart, to, toPart) {
    const out = { attn: 1.5, ff: 2.5, block: 3 };
    const inp = { block: 0, attn: 1, ff: 2 };
    return from * 4 + out[fromPart || 'block'] < to * 4 + inp[toPart || 'block'];
}

/** A jumper wire: the output of one block (or its attention/feed-forward)
 *  fed into the input of another. */
export function jumperModule(from, to, fromPart = 'block', toPart = 'block') {
    return {
        id: `p.${jumperPadId(from, to, fromPart, toPart)}`,
        enabled: true,
        target: { stage: 'dit', domain: 'structure', blocks: null },
        // 'match': the signal arrives at the level the destination expects
        // (levels inside a block differ by up to ~270x; unmatched wires
        // mostly blow the model up into noise).
        structure: { type: 'jumper', from, to, from_part: fromPart, to_part: toPart, level: 'match' },
        mix: 1.0,
        steps: { from: 0.0, to: 1.0 },
    };
}

/**
 * Every contact on the board, as columns in signal order. Each column is
 * one place on the path (an embedding, a DiT block, the latent, a decoder
 * block) with up to three pads; ids are stable strings that key the
 * user's names in the board map.
 */
export function boardColumns(registry) {
    const cols = [];
    const stages = STAGE_ORDER
        .map(s => (registry?.stages || []).find(x => x.stage === s))
        .filter(Boolean);
    for (const st of stages) {
        const add = (key, pads) => cols.push({ key, stage: st.stage, pads });
        if (st.stage === 'cond' || st.stage === 'timestep') {
            const base = st.stage === 'cond' ? 'cond' : 'time';
            add(base, [
                { id: base, stage: st.stage, domain: 'activation' },
                { id: `${base}.w`, stage: st.stage, domain: 'weight' },
            ]);
        } else if (st.stage === 'dit') {
            for (let i = 0; i < (st.count || 0); i++) {
                add(`dit.${i}`, [
                    { id: `dit.${i}`, stage: 'dit', domain: 'activation', block: i, part: 'block' },
                    { id: `dit.${i}.ff`, stage: 'dit', domain: 'weight', block: i, slot: 'ff', part: 'ff' },
                    { id: `dit.${i}.at`, stage: 'dit', domain: 'weight', block: i, slot: 'self_attn', part: 'attn' },
                ]);
            }
        } else if (st.stage === 'latent') {
            add('latent', [0, 1, 2].map(w => (
                { id: `latent.${w}`, stage: 'latent', domain: 'latent', window: w })));
        } else if (st.stage === 'decoder') {
            for (let j = 0; j < (st.count || 0); j++) {
                add(`vae.${j}`, [
                    { id: `vae.${j}`, stage: 'decoder', domain: 'activation', block: j },
                    { id: `vae.${j}.w`, stage: 'decoder', domain: 'weight', block: j },
                ]);
            }
        }
    }
    return cols;
}

// The weights of the two embedding layers drive the scale, shift and gate of
// every DiT block, so these contacts are hot: measured on the real model,
// cond.w blows the model up into noise above ~0.4 mix and time.w above ~0.6,
// even with the level kept, while ~0.3 bends audibly and keeps the music.
// They carry a built-in series resistor: their strength is this fraction of
// the board's dry/wet.
const HOT_CONTACTS = new Set(['cond.w', 'time.w']);
const HOT_CONTACT_MIX = 0.35;

/** What touching `pad` does — always the same for the same contact. */
export function padModule(pad, registry) {
    const rng = seededRandom(hashString(pad.id));
    const opDomain = pad.domain === 'latent' ? 'latent' : pad.domain;
    const ops = (registry?.operators || []).filter(o => o.domains.includes(opDomain));
    const op = pick(ops, rng);
    const mod = {
        id: `p.${pad.id}`,
        enabled: true,
        target: { stage: pad.stage, domain: pad.domain },
        operator: op?.name || 'scale',
        params: randomParams(op, 0.85, rng),
        mix: HOT_CONTACTS.has(pad.id) ? HOT_CONTACT_MIX : 1.0,
        // Contacts reshape the signal (or weights), never its energy: at full
        // strength an unlevelled bend on the small embedding layers that feed
        // every block blows the model up into noise.
        keep_level: true,
    };
    if (pad.block !== undefined) mod.target.blocks = [pad.block];
    if (pad.domain === 'weight') {
        mod.target.param = 'weight';
        if (pad.slot) mod.target.weight_slot = pad.slot;
    }
    if (rng() < 0.35) {
        mod.target.features = {
            mode: 'random', fraction: rand(0.2, 0.8, rng), seed: Math.floor(rng() * 9999),
        };
    }
    if (pad.stage === 'latent') {
        // Three contacts on the latent: early, middle and late in the run.
        mod.steps = { from: pad.window / 3, to: (pad.window + 1) / 3 };
    } else if (pad.stage === 'dit' && pad.domain === 'activation') {
        const a = rng() < 0.4 ? rand(0, 0.6, rng) : 0;
        mod.steps = { from: a, to: a ? Math.min(1, a + 0.4) : 1 };
    }
    return mod;
}

/** The modules for a set of touched/held pads and wires, at a dry/wet. */
export function probeModules(padIds, columns, registry, wet) {
    const byId = {};
    columns.forEach(c => c.pads.forEach(p => { byId[p.id] = p; }));
    return padIds.map((id) => {
        const j = parseJumperId(id);
        if (j && isNoopWire(j.from, j.fromPart, j.to, j.toPart)) return null;
        const mod = j ? jumperModule(j.from, j.to, j.fromPart, j.toPart)
            : byId[id] ? padModule(byId[id], registry) : null;
        return mod && { ...mod, mix: clamp((mod.mix ?? 1) * wet, 0, 1) };
    }).filter(Boolean);
}
