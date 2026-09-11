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
    break: { modules: [2, 3], mixRange: [0.8, 1.0], paramSpread: 1.0,  weightChance: 0.4,  structureChance: 0.3 },
};

const rand = (lo, hi) => lo + Math.random() * (hi - lo);
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

function randomParams(opSpec, spread) {
    const out = {};
    for (const [name, desc] of Object.entries(opSpec?.params || {})) {
        if (desc.type === 'bool') {
            out[name] = Math.random() < 0.3;
        } else if (desc.type === 'enum') {
            out[name] = Math.random() < 0.5 ? desc.default : pick(desc.options || [desc.default]);
        } else if (desc.type === 'curve') {
            out[name] = Array.from({ length: 6 }, () => rand(0, 2));
        } else if (desc.min !== undefined && desc.max !== undefined) {
            // Blend between the default and a fully random value by spread.
            const rnd = rand(desc.min, desc.max);
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
            const type = pick(['bypass', 'repeat', 'swap_nonlinearity']);
            mod.structure = { type };
            if (type === 'repeat') mod.structure.times = pick([2, 3]);
            if (type === 'swap_nonlinearity') mod.structure.fn = pick(registry?.swap_fns || ['sin']);
            const n = stageInfo.count || 8;
            const b = Math.floor(rand(0, n));
            mod.target.blocks = [b, Math.min(n - 1, b + 1)];
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
