# Training Rewire — SA3 LoRA on Projects

**Branch:** `dev/sa3`
**Status:** scoped, not started
**Companion to:** [DATASET_PREP_REDESIGN.md](DATASET_PREP_REDESIGN.md)
**Triggered by:** Phase 5a deleted the legacy 3-panel dataset prep, leaving training as the last consumer of `data/` + `data/metadata.json`.

---

## 0. TL;DR

Training trains on a **project** (the Dataset Workbench artifact) instead of an implicit `data/` directory. The user picks a project from a dropdown, hits Start, and `<projects_dir>/<name>/` (audio + `.txt` sidecars) becomes the dataset directly — no metadata.json read, no caption materialization, no `data/` writes.

Rename `app/core/training/fine_tuner.py` → `app/core/training/sa3_trainer.py`. Rename class `FineTuner` → `SA3Trainer`. The orchestrator stays — only the legacy *staging* path inside it goes. `sa3_lora_runner.py` keeps its name; it's the SA3-specific subprocess helper layer and that name is accurate.

Once the rewire lands, `data/`, `metadata.json`, `dataset-config.json`, `_stage_dataset`'s materialization branch, `materialize_captions()`, and the deprecated config helpers all get deleted as a single chunk.

---

## 1. Why we're doing this

Phase 5a removed the only UIs that wrote to `data/metadata.json` (`BulkAnnotatePanel`, `CsvImportPanel`, `process-files`). Training is now the only thing still *reading* it. Two pipelines for "where the dataset lives" is exactly the kind of two-paths-for-one-thing the Phase 1 redesign was trying to delete.

The new project workflow already produces SA3's canonical on-disk form: a folder with `<clip>.wav` + `<clip>.txt` sidecar pairs. Training just needs to point at it. Everything between (`metadata.json`, `materialize_captions`, `_stage_dataset`'s validation song-and-dance) is busywork left over from the era when prompts were stored in a JSON column.

The rename is a separate but related cleanup: `fine_tuner.py` reads like generic ML scaffolding, but the file is specifically the orchestrator for `vendor/stable-audio-3/scripts/train_lora.py`. The name should say that.

---

## 2. Mental model

```
            ┌─────────────────────────────────────────────┐
            │   PROJECT (on disk)                         │
            │   <projects_dir>/<name>/                    │
            │     pad1.wav  pad1.txt                      │
            │     kick.wav  kick.txt                      │
            │     .project.json                           │
            └─────────────────┬───────────────────────────┘
                              │  (user committed via "Create Dataset")
                              ▼
                  ┌──────────────────────────┐
                  │  Training tab            │
                  │  - pick project          │
                  │  - pick base model       │
                  │  - hyperparams           │
                  │  - Start                 │
                  └─────────────┬────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │  SA3Trainer (orchestr.)  │
                  │  reads project_path(name)│
                  │  no metadata.json        │
                  │  no .txt rewrites        │
                  └─────────────┬────────────┘
                                │
                                ▼
                  vendor/.../train_lora.py
                  --data_dir = project_path
```

The project folder IS the dataset. Training doesn't transform it; it just feeds it to the SA3 subprocess. The user's "commit" was the materialization step.

---

## 3. Training-tab UI changes

### 3.1 New: Dataset picker

Above the existing Mode toggle, a new compact row:

```
Dataset:  [▼ my_first_track (187 clips)]
```

- `Select` populated from `GET /api/projects` (already exists). Each item shows `name (N clips)`.
- Disabled when no projects exist; shows inline link to the Dataset tab.
- Persisted to `localStorage` (`fragmenta.training.lastProject`), defaulted to whatever was last loaded in the Dataset tab if available.
- If a previously selected project has been deleted, the picker falls back to "(none)" and the Start button is disabled.

### 3.2 Removed: dataset-status sticky card

The right-pane "Dataset Status" Paper that was driven by `/api/status` (raw file count, total duration, "has metadata.json") was already deleted from the Data Processing tab in Phase 5a — but [App.js](app/frontend/src/App.js) still passes `systemStatus` into the TrainingMonitor (line ~1683). That prop goes away.

Replaced by **inline project summary** below the dataset picker: `"187 clips · 1h 14m total · ✓ all annotated"` — pulled from the project's own `GET /api/projects/<name>/health` response (which already exists). No new endpoint.

### 3.3 Hyperparam suggester (existing button)

"Suggest hyperparameters" already calls `GET /api/training/suggest-hyperparams?mode=...`. The endpoint currently passes `config.get_path("data")` to the suggester. Rewire to accept a `project_name` query param and resolve via `project_path(name)`.

### 3.4 Start button gating

Disabled unless: project selected AND project has ≥ 1 committed clip AND base model is downloaded (last check already exists).

---

## 4. API surface

### 4.1 Changed contracts

| Endpoint | Method | Change |
|---|---|---|
| `/api/start-training` | POST | Body gains required `project_name`. `dataDir` / `data_dir` / any path override is removed from the schema. |
| `/api/training/suggest-hyperparams` | GET | Adds required query param `project_name`. Internal: passes `project_path(name)` to the suggester instead of `config.get_path("data")`. |

### 4.2 Endpoints that go away

| Endpoint | Why |
|---|---|
| `/api/status` | The only non-training reader of `data/metadata.json`. Replaced by per-project endpoints that already exist (`GET /api/projects`, `GET /api/projects/<name>`, `.../health`). |
| `/api/start-fresh` | Currently nukes `data/` + `metadata.json` + duration cache. With `data/` retired, the only remaining duty is resetting in-process generation state — which can be a frontend-only soft reset. The route + dialog stay only if we explicitly need a "panic clear" button; otherwise drop both. |

### 4.3 Unchanged

`/api/training/status`, `/api/stop-training`, `/api/training/checkpoint-preview`, `/api/loras`, `/api/unwrap-model`. None of these touch `data/`.

---

## 5. Disk layout after rewire

```
<user_data_dir>/
  projects/                      ← only source of training data
    my_first_track/
      pad1.wav  pad1.txt
      kick.wav  kick.txt
      .project.json
  models/
    pretrained/
      sa3/hub/                   ← SA3 base models (HF cache)
      clap/                      ← CLAP weights + text encoders
    fine_tuned/                  ← LoRA checkpoints written here
      my_lora/
        ...

(deleted)
  data/                          ← gone, with metadata.json
  models/config/dataset-config.json   ← gone
```

The `data/` directory is removed entirely. Anyone with existing `data/` content should manually move it into a project (or, as an optional kindness, we can ship a one-time **Import data/ as a project** action on first launch — Phase 5 of the dataset redesign already lists this).

---

## 6. Kill list — every legacy reference

Compiled from a full audit. Each item is either *delete* or *rewrite*.

### 6.1 Backend Python

**`app/core/training/fine_tuner.py`** (file is renamed to `sa3_trainer.py`)
- Lines 177–200 — `_stage_dataset()` body (reads `dataDir`/`data` from config, validates `metadata.json` exists, calls `materialize_captions`). **Rewrite** to: `project_path(name)` resolution + assert at least one `.txt` sidecar exists. ~10 lines instead of ~25.
- Class `FineTuner` (lines 81–392) → **rename** to `SA3Trainer`.
- Import of `materialize_captions` from `sa3_lora_runner` (line 57) → **delete**.
- Docstring at line 1 already says "SA3 LoRA training orchestrator" — keep, polish wording for rename.

**`app/core/training/sa3_lora_runner.py`**
- `materialize_captions()` function (lines 32–74) → **delete**. Sole caller is fine_tuner.

**`app/backend/app.py`**
- `/api/start-training` (lines 197–257):
  - Add `project_name` validation; reject if missing.
  - Pass `project_name` through to `start_training_func()`.
- `/api/training/suggest-hyperparams` (lines 749–764):
  - Replace `config.get_path('data')` (line 760) with `project_path(request.args.get('project_name'))`.
- `/api/status` (lines 660–746): **delete**. (Or replace with a trivial server-version probe if anything still needs it; verify no other callers first.)
- `/api/start-fresh` (lines 1226–1265): **delete or trim**. Currently clears `data/`, `metadata.json`, the duration cache, and generation state. Without `data/`, the route only has soft state left to clear, and that can be a frontend-only operation. Default to delete; reinstate as a soft-reset if a real user need surfaces.
- Imports of `start_training`, `get_training_status`, `stop_training`, `preview_training_plan` (line ~6) — update if the module rename happens.

**`app/core/config.py`**
- `get_path("data")` resolution (line ~77) → **delete the `"data"` case** from the path map. If something still calls it, that's a bug we should fail loudly on.
- `get_metadata_json_path()` (lines 97–98) → **delete**.
- `get_dataset_config_path()` (lines 88–91) → **delete**. Already a deprecated no-op.
- `get_custom_metadata_path()` (lines 93–95) → **delete**. Already returns `""`.
- `update_dataset_config()` (lines 100–104) → **delete**. Already a no-op.
- All `SA3-TODO(Phase 5)` comments in the same region — delete with the methods.

**`app/core/training/hyperparam_suggester.py`**
- `suggest(data_dir, mode)` (line ~194): keep the function signature, but it now receives a project folder path. No code change needed inside `suggest` itself — it already globs audio files; it doesn't care whether the folder is `data/` or a project.
- `.duration_cache.json` (created inside `data_dir`, line ~24, 204): now lives inside the project folder. Project's `Discard` already wipes everything inside the project, so the cache gets the right lifecycle for free. Consider stashing the cache under `<project>/.duration_cache.json` (hidden file) so it doesn't pollute the user-visible folder.

### 6.2 Frontend

**`app/frontend/src/App.js`**
- `systemStatus` state (line ~229) + `isStatusLoading` (line ~230) → **delete**. Used only for the now-gone Dataset Status panel + the prop currently passed into TrainingMonitor.
- `fetchSystemStatus()` (lines ~374–384) + every caller (~lines 474, 901) → **delete**.
- Add `trainingProject` state + `trainingProjects` list + a `Select` in the training-tab JSX (currently around lines 1235–1292 for the base-model dropdown — new dataset picker goes immediately above).
- `startTraining()` (lines ~614–665): include `project_name: trainingProject` in the POST body; drop any `dataDir` field that survived.
- `fetchHyperparamSuggestion()` (lines ~592–610): append `&project_name=...` to the URL.
- Training tab's right-pane TrainingMonitor (line ~1683): stop passing `systemStatus`; replace inline summary with the project-folder summary above.

**`app/frontend/src/components/TrainingMonitor.js`**
- Remove `systemStatus` from its props. Anything that uses it (dataset overview) belongs in App.js's new inline project summary block, not in the monitor.

**`app/frontend/src/components/LoraCheckpointManager.js`**
- No changes. It already operates on `models/fine_tuned/` and is project-agnostic.

### 6.3 Configuration / artifacts

- `models/config/dataset-config.json` — **delete the file** if it exists on disk. Already a no-op.
- Any `data/metadata.json` and `data/.duration_cache.json` from prior runs — left to the user; we're not migrating automatically.

---

## 7. Phased rollout

### Phase A — Project-aware training start

1. Add `project_name` to `/api/start-training` (validation, plumbing to `SA3Trainer.start()`).
2. Rewrite `_stage_dataset` to use `project_path(name)`; verify .txt sidecars exist.
3. Frontend training tab: add the dataset picker, persist `lastProject`, gate Start.
4. `fetchHyperparamSuggestion` sends `project_name`.

Ship after this is testable end-to-end with both old and new endpoints coexisting.

### Phase B — Hyperparam suggester

1. `/api/training/suggest-hyperparams` accepts and requires `project_name`.
2. Cache file moves inside project folder.

### Phase C — Rename

1. `git mv app/core/training/fine_tuner.py app/core/training/sa3_trainer.py`.
2. Class `FineTuner` → `SA3Trainer`. Internal references updated.
3. Update imports in `app/backend/app.py`.
4. Update the docstring + module-level references.

### Phase D — Delete legacy paths

1. Delete `_stage_dataset`'s legacy branch (now unreachable).
2. Delete `materialize_captions` from `sa3_lora_runner.py`.
3. Delete `/api/status`, `/api/start-fresh` (or trim — see §4.2).
4. Delete `systemStatus` / `fetchSystemStatus` / `isStatusLoading` from App.js.
5. Delete `config.get_path("data")`, `get_metadata_json_path`, `get_dataset_config_path`, `get_custom_metadata_path`, `update_dataset_config`, the corresponding SA3-TODO comments.
6. Delete `models/config/dataset-config.json` file if present.

### Phase E — Cleanups (low priority)

- `_safe_name` in fine_tuner: re-audit, ensure it handles project names with spaces / special characters identically to the dataset workbench's `sanitize_project_name`.
- TrainingMonitor: review whether its current LossChart + step display still surfaces everything we want now that the dataset-status portion is gone. May want to add elapsed/eta.

---

## 8. Known limits / open questions

- **What if the user starts training, then immediately edits the source project?** The training subprocess holds open file handles to the audio files it's about to read. With `data/` we had a defensive `materialize_captions` step that essentially snapshotted prompts at start. Now, mid-training prompt edits + saves would be ignored by the running subprocess (it already read the sidecars). That's fine — the existing behavior is what users expect — but worth documenting in the training tab: *"Edits to the project mid-run only affect future runs."*

- **Existing `data/` content.** Phase D deletes the directory unconditionally. Anyone with audio in `data/` who hasn't moved it to a project loses it. Mitigations:
  1. Soft delete: rename to `data.legacy/` instead of removing, surface a one-time toast.
  2. Migration helper: a "Import `data/` as a project" button in the Dataset tab (already on the Phase 5 backlog in [DATASET_PREP_REDESIGN.md](DATASET_PREP_REDESIGN.md) §7).
  3. Just delete and document.

  Recommend (2). Cheap to write; respects user data.

- **Hyperparam suggester runs on the project folder including drafts.** If `lastProject` has uncommitted clips, the suggester sees them. That's defensible — the user is suggesting hyperparams for the dataset *as they're shaping it*. But the actual training will use whatever was committed; brief mismatch possible. Surface in the tooltip.

- **`/api/start-fresh` survival.** Currently does FIVE things: clear `data/`, clear metadata.json, clear duration cache, clear performance session, wipe MIDI mappings. Of those, only the last two are still meaningful. Either trim the route to those, or kill the route + bind the dialog to a frontend-only soft reset. Either's fine; lean toward kill.

- **Where do LoRA artifacts live in this world?** Already at `models/fine_tuned/<name>/`. Not project-coupled — a single project can yield many LoRAs. That's correct.

- **Multi-project training.** Not in scope. If we ever want to train on a union of projects, the dataset picker becomes a multi-select. Don't design for this yet.

---

## 9. Notes for future implementers

- The SA3 subprocess takes `--data_dir`, which it then passes to `SampleDataset`. `SampleDataset` does its own audio file enumeration; it doesn't read metadata. The project folder layout (audio + sibling `.txt`) IS what it expects. There's nothing to translate.
- `sa3_lora_runner.build_train_command` already takes `data_dir: Path` — no signature change there.
- `prestage_base_model` is HF-cache work, not dataset work. Don't touch.
- The "Suggest hyperparameters" feature is the only place outside training itself that needs to know about the project's audio. Everything else (project selection, base-model selection, hyperparam tuning) is metadata + UI.
- TrainingMonitor's loss-curve is parsed from `<run_dir>/csv/version_0/metrics.csv` written by Lightning. Not project-related; ignore during this rewire.
- Tests: no tests currently cover these endpoints. Adding contract tests for `/api/start-training` with `project_name` would be cheap insurance once Phase A lands.

---

## 10. Naming agreed

`fine_tuner.py` → `sa3_trainer.py`. Class `FineTuner` → `SA3Trainer`. Singleton getter `get_trainer()` → keep the name. Module-level public functions (`start_training`, `get_training_status`, `stop_training`, `preview_training_plan`) — names accurate, keep.

`sa3_lora_runner.py` keeps its name. It's a *runner* (subprocess-side helpers), not a *trainer* (lifecycle), and the distinction matters once we want a second runner backend in the future.
