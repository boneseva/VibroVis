/**
 * url_reconcile.js: applies the view described by a valid incoming URL to the running app (Phase 2.2 of URL sharing).
 *
 * Division of labour
 *   url_restore.js   decides WHETHER there is something to restore (state machine -> public `url-init` Store).
 *   url_plan.py      validates the filter values against this browser's own dataset (server side, pure).
 *   this file        WRITES the validated values into the widgets, stage by stage, and proves they stuck.
 *
 * How it stays correct in a callback cascade
 *   * It never uses callback Outputs: widgets are written with dash_clientside.set_props only (an Output write desyncs
 *     the rendered Dropdown; set_props does not). It never writes a value that already matches (compare-and-set).
 *   * A stage advances only when the app has gone QUIET: no Dash callback request in flight and none started for
 *     `settleMs`. That is exactly when every reset the previous write provoked (a new location resets the model, a new
 *     dataset resets every filter, ...) has landed, so a default can never overwrite a restored value afterwards.
 *   * Every write is verified on the next quiet tick (actual prop == wanted). A mismatch is rewritten (at most
 *     `maxRewrites` times, then the value is dropped as `unstable`). After the last key a full verification pass
 *     repeats until nothing needs a write, so a late reset after apparent convergence is corrected before completion.
 *
 * Stages: identity (project -> location -> model -> k, validated against the live option lists)
 *         -> dataset (wait for the browser's own dataset) -> plan (server) -> filters (STAGE_KEYS order) -> verify.
 * The data load holds (url_state.restore_should_hold) until identity is settled, so no default dataset is loaded in vain.
 *
 * A genuine user action on a restorable control (trusted DOM event) cancels the restore; whatever is already applied
 * stays, the rest is dropped as `interrupted`. Every cancellation also releases the load hold. Nothing here writes the
 * URL (that is the live-sync phase) or runs without a valid, active initialisation.
 */
(function (root, factory) {
  const api = factory(root);
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (root) root.VibrovisUrlReconcile = api;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function (root) {
  'use strict';

  const DEFAULTS = {
    settleMs: 400,         // quiet window: no request in flight and none started for this long
    pollMs: 100,
    maxRewrites: 3,        // per key, then dropped as `unstable`
    maxVerifyPasses: 6,
    noKeyPolls: 20,        // consecutive quiet polls without the expected dataset before `dataset_unavailable`
    planTimeoutMs: 30000,
    restoreMs: 180000,     // watchdog for the whole restoration
    syncDebounceMs: 700,   // live URL: the snapshot must stay unchanged this long
    shareWaitMs: 15000,    // Share waits this long for the view to settle
  };

  const IDENTITY = ['project', 'loc', 'model', 'k'];
  const ID_WIDGET = { project: 'project-dropdown', loc: 'location-dropdown', model: 'model-dropdown', k: 'num-cluster-dropdown' };
  const IDENTITY_REASON = { project: 'unknown_project', loc: 'unknown_location', model: 'unknown_model', k: 'unknown_k' };
  // Order matters (dependencies); mirrors url_state.STAGE_KEYS['filters'] (a test asserts equality).
  const FILTER_ORDER = ['view', 'color', 'annot', 'cmp', 'micro', 'rec', 'ch', 'dates', 'hours', 'clusters', 'hide',
    'labels', 'merge', 'mth', 'mcc', 'max', 'prio', 'sort'];

  // ---- small pure helpers --------------------------------------------------------------------------------------------
  const asArray = v => (Array.isArray(v) ? v : (v == null ? [] : [v]));
  function sameSet(a, b) {
    a = asArray(a); b = asArray(b);
    if (a.length !== b.length) return false;
    const s = new Set(a.map(String));
    return b.every(x => s.has(String(x)));
  }
  const sameNum = (a, b) => a != null && b != null && Math.abs(Number(a) - Number(b)) < 1e-9;
  function sameRange(a, b) {
    return Array.isArray(a) && Array.isArray(b) && a.length === 2 && b.length === 2 && sameNum(a[0], b[0]) && sameNum(a[1], b[1]);
  }
  function sameJson(a, b) { return JSON.stringify(a) === JSON.stringify(b); }

  /** Shared vectors (tests/fixtures/url_restore_vectors.json): mirror of url_state.labels_to_tick. */
  function labelsToTick(labels, rows) {
    rows = Array.from(rows);
    const others = rows.filter(r => r !== 'Unlabeled');
    if (labels === 'unlabeled') {
      if (!rows.includes('Unlabeled')) return { action: 'drop', ticked: [], reason: 'no_unlabeled_row' };
      if (!others.length) return { action: 'noop', ticked: ['Unlabeled'], reason: null };
      return { action: 'apply', ticked: ['Unlabeled'], reason: null };
    }
    if (labels === 'labeled') {
      if (!others.length) return { action: 'drop', ticked: [], reason: 'no_labeled_rows' };
      return { action: 'apply', ticked: others, reason: null };
    }
    return { action: 'drop', ticked: [], reason: 'unknown_value' };
  }

  /** Mirror of url_state.cluster_ticks. */
  function clusterTicks(clusters, hide, clusterIds) {
    const ids = Array.from(new Set(Array.from(clusterIds).map(Number))).sort((a, b) => a - b);
    let ticked;
    if (clusters != null) { const s = new Set(clusters.map(Number)); ticked = ids.filter(c => s.has(c)); }
    else if (hide != null) { const s = new Set(hide.map(Number)); ticked = ids.filter(c => !s.has(c)); }
    else return { action: 'noop', ticked: ids, reason: null };
    if (!ticked.length) return { action: 'drop', ticked: [], reason: 'empty_cluster_selection' };
    if (ticked.length === ids.length && ticked.every((c, i) => c === ids[i])) return { action: 'noop', ticked: ids, reason: null };
    return { action: 'apply', ticked, reason: null };
  }

  // ---- adapters: (wanted value, env) -> {writes: [{id, props} | {view}], drop?: reason, applied?: effective value} -----
  function simple(id, prop, eq) {
    return (value, env) => (eq(env.value(id, prop), value) ? { writes: [] } : { writes: [{ id, props: { [prop]: value } }] });
  }
  /**
   * A slider value that must not exceed the slider's REAL maximum at apply time (the clip-count maximum only exists once
   * merging is on and the figure has settled, so the server cannot know it). The clamp is reported as an adjustment.
   */
  function clamped(id) {
    const base = simple(id, 'value', sameNum);
    return (value, env) => {
      const max = env.value(id, 'max');
      const eff = (typeof max === 'number' && isFinite(max) && value > max) ? max : value;
      const res = base(eff, env);
      if (eff !== value) res.adjust = { value, applied: eff, reason: 'clamped' };
      return res;
    };
  }
  function rowWrites(rows, wantedIds, type) {
    const want = new Set(wantedIds.map(String));
    const writes = [];
    for (const r of rows) {
      const on = asArray(r.value).includes('on');
      const should = want.has(String(r.id.index));
      if (on !== should) writes.push({ id: r.id, props: { value: should ? ['on'] : [] } });
    }
    return writes;
  }

  const ADAPTERS = {
    view: (v, env) => (env.currentView() === v ? { writes: [] } : { writes: [{ view: v }] }),
    color: simple('color-mode-radio', 'value', (a, b) => a === b),
    annot: simple('raven-annotator-filter', 'value', sameSet),
    cmp: simple('raven-compare-mode', 'value', (a, b) => a === b),
    micro: simple('microlocation-dropdown', 'value', sameSet),
    rec: simple('recorder-type-dropdown', 'value', sameSet),
    ch: simple('channel-checklist', 'value', sameSet),
    hours: simple('hour-slider', 'value', sameRange),
    merge: simple('merge-switch', 'on', (a, b) => !!a === !!b),
    mth: clamped('merge-threshold'),
    mcc: clamped('clip-count-threshold'),
    max: simple('max-points', 'value', sameNum),
    prio: (v, env) => {
      const want = v ? ['on'] : [];
      return sameSet(env.value('priority-sampling-toggle', 'value'), want) ? { writes: [] }
        : { writes: [{ id: 'priority-sampling-toggle', props: { value: want } }] };
    },
    sort: (v, env) => {
      const want = { col: v[0], asc: !!v[1] };
      if (sameJson(env.value('table-sort-store', 'data'), want)) return { writes: [] };
      return { writes: [{ id: 'table-sort-store', props: { data: want } }, { id: 'table-page-store', props: { data: 0 } }] };
    },
    dates: (v, env) => {
      const wanted = new Set(v.map(String));
      const simpleDd = env.rows('simple-date-dropdown');
      const tree = env.rows('date-checklist');
      const comps = simpleDd.length ? simpleDd : tree;
      if (!comps.length) return { writes: [], drop: 'no_date_widget' };
      const writes = [];
      for (const c of comps) {
        const want = asArray(c.options).map(o => o.value).filter(x => wanted.has(String(x)));
        if (!sameSet(c.value, want)) writes.push({ id: c.id, props: { value: want } });
      }
      return { writes };
    },
    clusters: (v, env) => clusterAdapter(v, null, env),
    hide: (v, env) => clusterAdapter(null, v, env),
    labels: (v, env) => {
      const rows = env.rows('label-checkbox');
      const res = labelsToTick(v, rows.map(r => String(r.id.index)));
      if (res.action === 'drop') return { writes: [], drop: res.reason };
      return { writes: rowWrites(rows, res.ticked) };
    },
  };
  function clusterAdapter(clusters, hide, env) {
    const rows = env.rows('cluster-checkbox');
    const res = clusterTicks(clusters, hide, rows.map(r => r.id.index));
    if (res.action === 'drop') return { writes: [], drop: res.reason };
    return { writes: rowWrites(rows, res.ticked) };
  }

  // ---- run state ---------------------------------------------------------------------------------------------------
  function createRun(payload, now) {
    const desired = payload.state || {};
    return {
      token: payload.token, desired, warnings: payload.warnings || [], stage: 'identity', status: 'running', reason: null,
      idKeys: IDENTITY.filter(k => desired[k] !== undefined), idx: 0, pending: null, target: {}, writes: {},
      applied: {}, dropped: [], adjusted: [], startedAt: now, planId: null, planSince: null, plan: null,
      fKeys: null, fi: 0, verifyPasses: 0, noKey: 0, log: [],
    };
  }
  const note = (run, what) => { if (run.log.length < 400) run.log.push(what); };
  const drop = (run, key, value, reason) => { run.dropped.push({ key, value, reason }); note(run, `drop:${key}:${reason}`); };
  const isDone = run => run.status !== 'running';

  function perform(run, env, key, writes) {
    run.writes[key] = (run.writes[key] || 0) + 1;
    for (const w of writes) {
      if (w.view !== undefined) env.setView(w.view);
      else env.set(w.id, w.props);
    }
    env.markWrite();
    note(run, `write:${key}#${run.writes[key]}`);
  }

  function expectedKey(env) {
    return [env.value('project-dropdown', 'value'), env.value('model-dropdown', 'value'),
      asArray(env.value('location-dropdown', 'value')).map(String).sort()];
  }

  function finishIdentity(run, env) {
    env.set('url-identity', { data: { token: run.token, done: true, reason: run.reason, dropped: run.dropped.map(d => d.key) } });
    env.markWrite();
    note(run, 'identity_done');
    run.stage = Object.keys(run.desired).some(k => FILTER_ORDER.includes(k)) ? 'dataset' : 'finish';
  }

  function dropAllFilters(run, reason) {
    for (const k of FILTER_ORDER) if (run.desired[k] !== undefined && run.applied[k] === undefined
      && !run.dropped.some(d => d.key === k)) drop(run, k, run.desired[k], reason);
    run.stage = 'finish';
  }

  /** Returns true when this stage can proceed without the app being quiet (waiting on a server answer). */
  function needsQuiet(run) { return !(run.stage === 'plan' && run.planId); }

  function stepIdentity(run, env, cfg) {
    if (run.pending) {
      const k = run.pending, want = run.target[k];
      const eq = k === 'loc' ? sameSet : (a, b) => a === b || sameJson(a, b);
      if (eq(env.value(ID_WIDGET[k], 'value'), want)) { run.applied[k] = want; run.pending = null; run.idx++; return; }
      if ((run.writes[k] || 0) >= cfg.maxRewrites) { drop(run, k, want, 'unstable'); run.pending = null; run.idx++; return; }
      perform(run, env, k, [{ id: ID_WIDGET[k], props: { value: want } }]);
      return;
    }
    if (run.idx >= run.idKeys.length) return finishIdentity(run, env);
    const k = run.idKeys[run.idx], want = run.desired[k], widget = ID_WIDGET[k];
    const vals = asArray(env.options(widget)).map(o => o.value);
    let target;
    if (k === 'loc') {
      const kept = asArray(want).filter(x => vals.includes(x));
      if (!kept.length) drop(run, k, want, IDENTITY_REASON[k]);
      else { target = kept; asArray(want).filter(x => !vals.includes(x)).forEach(g => drop(run, k, g, IDENTITY_REASON[k])); }
    } else if (vals.includes(want)) target = want;
    else drop(run, k, want, IDENTITY_REASON[k]);
    if (target === undefined) { run.idx++; return; }
    run.target[k] = target;
    const cur = env.value(widget, 'value');
    if (k === 'loc' ? sameSet(cur, target) : cur === target) { run.applied[k] = target; run.idx++; return; }
    perform(run, env, k, [{ id: widget, props: { value: target } }]);
    run.pending = k;
  }

  function stepDataset(run, env, cfg) {
    const meta = env.value('derived-model-metadata', 'data');
    const key = meta && meta.key, want = expectedKey(env);
    if (key && key[0] === want[0] && key[1] === want[1] && sameSet(key[2], want[2])) { run.stage = 'plan'; run.noKey = 0; return; }
    if (++run.noKey >= cfg.noKeyPolls) dropAllFilters(run, 'dataset_unavailable');
  }

  function stepPlan(run, env, cfg, now) {
    if (!run.planId) {
      run.planId = env.uid(); run.planSince = now;
      env.set('url-plan-request', { data: {
        id: run.planId, token: run.token, key: expectedKey(env), k: env.value('num-cluster-dropdown', 'value'),
        merge_max: env.value('merge-threshold', 'max'), clip_max: env.value('clip-count-threshold', 'max') } });
      env.markWrite();
      return;
    }
    const plan = env.value('url-plan-filters', 'data');
    if (!plan || plan.id !== run.planId) {
      if (now - run.planSince > cfg.planTimeoutMs) dropAllFilters(run, 'plan_timeout');
      return;
    }
    if (plan.status !== 'ok' || plan.token !== run.token) { dropAllFilters(run, plan.status === 'ok' ? 'stale' : plan.status); return; }
    run.plan = plan;
    plan.dropped.forEach(d => run.dropped.push(d));
    plan.adjusted.forEach(a => run.adjusted.push(a));
    run.fKeys = FILTER_ORDER.filter(k => plan.applied[k] !== undefined);
    run.fi = 0; run.stage = run.fKeys.length ? 'filters' : 'finish';
  }

  function stepFilters(run, env, cfg) {
    if (run.fi >= run.fKeys.length) { run.stage = 'verify'; run.verifyPasses = 0; run.vi = 0; return; }
    const k = run.fKeys[run.fi], value = run.plan.applied[k];
    const res = ADAPTERS[k](value, env);
    if (res.adjust && !run.adjusted.some(a => a.key === k)) run.adjusted.push(Object.assign({ key: k }, res.adjust));
    if (res.drop) { drop(run, k, value, res.drop); run.pending = null; run.fi++; return; }
    if (!res.writes.length) { run.applied[k] = res.adjust ? res.adjust.applied : value; run.pending = null; run.fi++; return; }
    if ((run.writes[k] || 0) >= cfg.maxRewrites) { drop(run, k, value, 'unstable'); run.pending = null; run.fi++; return; }
    perform(run, env, k, res.writes);
  }

  function stepVerify(run, env, cfg) {
    // Re-check every key still applied; the first one that no longer matches is rewritten (bounded), then we re-check
    // from the start. A full clean pass means the view is stable.
    for (const k of run.fKeys) {
      if (run.applied[k] === undefined) continue;
      const res = ADAPTERS[k](run.plan.applied[k], env);
      if (res.drop) { delete run.applied[k]; drop(run, k, run.plan.applied[k], res.drop); return; }
      if (!res.writes.length) continue;
      if ((run.writes[k] || 0) >= cfg.maxRewrites || ++run.verifyPasses > cfg.maxVerifyPasses) {
        delete run.applied[k]; drop(run, k, run.plan.applied[k], 'unstable'); return;
      }
      note(run, `late_reset:${k}`);
      perform(run, env, k, res.writes);
      return;
    }
    run.stage = 'finish';
  }

  /**
   * A restore that wrote anything ends with ONE figure refresh. The scatter callback reads the legend rows when it is
   * dispatched; during a restore the dataset load and the rebuilt legend (new k / colour) can overlap, so the figure can
   * be computed from rows that are about to be replaced and nothing re-runs it afterwards (seen in a real browser).
   * The refresh re-writes `max-points` with the value it already has. That is an ordinary filter trigger of the scatter
   * callback: it recomputes from the settled widgets with the app's FIXED sampling seed (42), so the same state always
   * gives the same sampled rows. It deliberately does NOT go through the Resample button, whose clicks draw an unseeded
   * random sample (a restore must never change which points are shown). A restore that wrote nothing (idempotent
   * restart) refreshes nothing.
   */
  function stepFinish(run, env) {
    if (!run.refreshed && Object.keys(run.writes).length > 0) {
      run.refreshed = true;
      const max = env.value('max-points', 'value');
      if (typeof max === 'number' && isFinite(max)) {
        env.set('max-points', { value: max });
        env.markWrite();
        note(run, 'refresh_figure');
        return;                                        // the next quiet tick (after the figure callback) completes the run
      }
      note(run, 'refresh_skipped_no_max_points');
    }
    finish(run, env, 'complete');
  }

  function finish(run, env, status, reason) {
    run.status = status; run.reason = reason || null; run.stage = 'done';
    env.progress({ token: run.token, status, reason: run.reason, applied: run.applied, dropped: run.dropped,
      adjusted: run.adjusted, warnings: run.warnings, writes: run.writes, log: run.log.slice(0, 120) });
  }

  /** Stop restoring (user action, preset, watchdog, newer URL). Keeps what is applied; drops the rest as `interrupted`. */
  function cancel(run, env, reason) {
    if (isDone(run)) return;
    if (run.stage === 'identity') finishIdentity(run, env);       // always release the load hold
    for (const k of IDENTITY.concat(FILTER_ORDER)) {
      if (run.desired[k] !== undefined && run.applied[k] === undefined && !run.dropped.some(d => d.key === k)
        && !(run.plan && run.plan.dropped.some(d => d.key === k))) drop(run, k, run.desired[k], 'interrupted');
    }
    note(run, `cancel:${reason}`);
    finish(run, env, 'cancelled', reason);
  }

  /** One unit of work. Mutates *run*. *env*: see makeEnv (tests pass a fake). */
  function step(run, env, now, cfg) {
    cfg = Object.assign({}, DEFAULTS, cfg || {});
    if (isDone(run)) return run;
    if (now - run.startedAt > cfg.restoreMs) { cancel(run, env, 'timeout'); return run; }
    if (needsQuiet(run) && !env.quiet(now, cfg.settleMs)) return run;
    switch (run.stage) {
      case 'identity': stepIdentity(run, env, cfg); break;
      case 'dataset': stepDataset(run, env, cfg); break;
      case 'plan': stepPlan(run, env, cfg, now); break;
      case 'filters': stepFilters(run, env, cfg); break;
      case 'verify': stepVerify(run, env, cfg); break;
      case 'finish': stepFinish(run, env); break;
      default: break;
    }
    return run;
  }

  function noticeText(run, payloadWarnings) {
    const drops = (run && run.dropped) || [];
    const warns = (payloadWarnings || []).filter(w => w.code !== 'clamped');
    const n = drops.length + warns.length;
    if (!n) return '';
    const names = Array.from(new Set(drops.map(d => d.key).concat(warns.map(w => w.key)).filter(Boolean))).sort();
    return `${n} ${n === 1 ? 'setting' : 'settings'} in this link could not be applied` + (names.length ? ` (${names.join(', ')}).` : '.');
  }

  // ---- live URL and share: snapshot of the settled view (serialisation itself stays in Python) -------------------------
  /**
   * Raw widget values describing the current view. The server (url_plan.state_from_snapshot + url_state.rewrite_search)
   * turns it into a query string; nothing here knows the URL syntax. Transient state (hover, playback, loading, free text
   * such as label names) is never collected.
   */
  function collectSnapshot(env, search) {
    const ids = rows => rows.map(r => r.id.index);
    const ticked = rows => rows.filter(r => asArray(r.value).includes('on')).map(r => r.id.index);
    const cl = env.rows('cluster-checkbox'), lb = env.rows('label-checkbox');
    return {
      key: expectedKey(env),
      project: env.value('project-dropdown', 'value'), loc: env.value('location-dropdown', 'value'),
      model: env.value('model-dropdown', 'value'), k: env.value('num-cluster-dropdown', 'value'),
      view: env.value('main-view-tabs', 'data'), color: env.value('color-mode-radio', 'value'),
      annot: env.value('raven-annotator-filter', 'value'), cmp: env.value('raven-compare-mode', 'value'),
      micro: env.value('microlocation-dropdown', 'value'), rec: env.value('recorder-type-dropdown', 'value'),
      ch: env.value('channel-checklist', 'value'), dates: env.value('date-dropdown', 'data'),
      hours: env.value('hour-slider', 'value'),
      clusters: { ids: ids(cl), ticked: ticked(cl) }, labels: { rows: ids(lb).map(String), ticked: ticked(lb).map(String) },
      merge: env.value('merge-switch', 'on'), mth: env.value('merge-threshold', 'value'),
      mcc: env.value('clip-count-threshold', 'value'), max: env.value('max-points', 'value'),
      prio: env.value('priority-sampling-toggle', 'value'), sort: env.value('table-sort-store', 'data'),
      search: search || '',
    };
  }
  const canon = snap => { const c = Object.assign({}, snap); delete c.search; return JSON.stringify(c); };

  /**
   * Live URL decision (pure). *ss* = {lastSent, pendingCanon, pendingSince}; *ctx* = {decided, restoring, quiet, loaded}.
   * A snapshot is sent only when it is SETTLED (initialisation decided, no restore running, the app quiet, the dataset
   * the widgets name actually loaded) and has been unchanged for `debounceMs`. Returns 'request' or 'none'.
   */
  function syncDecide(ss, snap, ctx, now, debounceMs) {
    if (!ctx.decided || ctx.restoring || !ctx.quiet || !ctx.loaded) { ss.pendingCanon = null; return 'none'; }
    const c = canon(snap);
    if (c === ss.lastSent) { ss.pendingCanon = null; return 'none'; }
    if (c !== ss.pendingCanon) { ss.pendingCanon = c; ss.pendingSince = now; return 'none'; }
    if (now - ss.pendingSince < debounceMs) return 'none';
    ss.lastSent = c;
    return 'request';
  }

  // ---- Dash binding (browser only) ---------------------------------------------------------------------------------------
  const CONTROL_IDS = ['project-dropdown', 'location-dropdown', 'location-map-container', 'location-map-modal', 'model-dropdown',
    'num-cluster-dropdown', 'color-mode-radio', 'raven-annotator-filter', 'raven-compare-mode', 'microlocation-dropdown',
    'recorder-type-dropdown', 'channel-checklist', 'all-or-none-channel', 'date-tree-container', 'hour-slider',
    'all-or-none-cluster', 'all-or-none-label', 'max-points', 'resample-btn', 'priority-sampling-toggle', 'merge-switch',
    'merge-threshold', 'clip-count-threshold', 'tab-btn-scatter', 'tab-btn-table', 'cluster-list-container',
    'label-list-container'];
  const PRESET_IDS = ['preset-load-btn'];
  const CONTROL_TYPES = ['cluster-checkbox', 'label-checkbox', 'raven-checkbox', 'date-checklist', 'month-select-all',
    'year-select-all', 'simple-date-dropdown', 'table-header'];
  const IGNORED_KEYS = ['Tab', 'Shift', 'Alt', 'Escape', 'Control', 'Meta'];

  /** Pure: which registered control (if any) does this DOM target belong to? 'preset' | 'control' | null. */
  function classifyTarget(el, doc) {
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const id = n.id;
      if (!id) continue;
      if (PRESET_IDS.includes(id)) return 'preset';
      if (CONTROL_IDS.includes(id)) return 'control';
      if (id.charAt(0) === '{') {
        try { if (CONTROL_TYPES.includes(JSON.parse(id).type)) return 'control'; } catch (err) { /* not a pattern id */ }
      }
    }
    return null;
  }
  /** Pure: does this keyboard event count as the user deliberately operating a control? */
  function countsAsKey(e) {
    if (e.ctrlKey || e.metaKey) return false;
    return !IGNORED_KEYS.includes(e.key);
  }

  const net = { inflight: 0, last: 0 };
  function installNetworkProbe() {
    if (!root.fetch || root.fetch.__vvProbe) return;
    const orig = root.fetch.bind(root);
    const nowMs = () => (root.performance ? root.performance.now() : Date.now());
    const wrapped = function (input, init) {
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      const tracked = url.indexOf('_dash-update-component') >= 0;
      if (tracked) { net.inflight++; net.last = nowMs(); }
      const p = orig(input, init);
      if (tracked) {
        const done = () => { net.inflight = Math.max(0, net.inflight - 1); net.last = nowMs(); };
        p.then(done, done);
      }
      return p;
    };
    wrapped.__vvProbe = true;
    root.fetch = wrapped;
  }

  function makeEnv() {
    const dc = root.dash_clientside, api = root.dash_component_api;
    const nowMs = () => (root.performance ? root.performance.now() : Date.now());
    let lastWrite = 0, uidN = 0;
    const layout = id => { try { return api.getLayout(id); } catch (err) { return undefined; } };
    const env = {
      value: (id, prop) => { const l = layout(id); return l && l.props ? l.props[prop === undefined ? 'data' : prop] : undefined; },
      options: id => { const l = layout(id); return l && l.props ? l.props.options : []; },
      set: (id, props) => dc.set_props(id, props),
      markWrite: () => { lastWrite = nowMs(); },
      quiet: (now, settleMs) => net.inflight === 0 && now - Math.max(net.last, lastWrite) >= settleMs,
      uid: () => `${Date.now().toString(36)}-${(++uidN)}-${Math.random().toString(36).slice(2, 8)}`,
      rows: type => {
        const out = [];
        const els = root.document.querySelectorAll('[id^="{"]');
        for (const el of els) {
          let id;
          try { id = JSON.parse(el.id); } catch (err) { continue; }
          if (id.type !== type) continue;
          const l = layout(id);
          if (l && l.props) out.push({ id, value: l.props.value, options: l.props.options });
        }
        return out;
      },
      currentView: () => (env.value('main-view-tabs', 'data') === 'table-tab' ? 'table' : 'scatter'),
      // Same DOM effect as the existing tab callback (tests assert the DOM equals a real click on the tab button).
      setView: view => {
        const d = root.document, isTable = view === 'table';
        const sp = d.getElementById('scatter-pane'), tp = d.getElementById('table-pane');
        const bs = d.getElementById('tab-btn-scatter'), bt = d.getElementById('tab-btn-table');
        if (sp && tp && bs && bt) {
          sp.classList.toggle('view-pane--active', !isTable); tp.classList.toggle('view-pane--active', isTable);
          bs.classList.toggle('view-tab-btn--active', !isTable); bt.classList.toggle('view-tab-btn--active', isTable);
        }
        dc.set_props('main-view-tabs', { data: isTable ? 'table-tab' : 'scatter-tab' });
      },
      progress: info => {
        dc.set_props('url-progress', { data: info });
        const text = noticeText(info, info.warnings);
        showNotice(text);
      },
    };
    return env;
  }

  function showNotice(text) {
    try {
      root.dash_clientside.set_props('url-restore-notice-text', { children: text });
      root.dash_clientside.set_props('url-restore-notice', { style: noticeStyle(!!text) });
    } catch (err) { /* the notice is optional */ }
  }
  function noticeStyle(show) {
    return { display: show ? 'flex' : 'none', position: 'fixed', left: '12px', bottom: '12px', zIndex: 2500, maxWidth: '420px',
      alignItems: 'center', gap: '8px', background: '#fff3cd', border: '1px solid #e0c36a', borderRadius: '6px',
      padding: '8px 10px', fontSize: '0.85em', boxShadow: '0 2px 8px rgba(0,0,0,0.15)' };
  }

  const S = { run: null, env: null, cfg: DEFAULTS, timer: null, lastToken: null, lastNotice: null, log: [],
    sync: { lastSent: null, pendingCanon: null, pendingSince: 0, reqId: null, lastResultId: null, applied: [] }, share: null };

  function readConfig(env) {
    const c = env.value('url-init-config', 'data') || {};
    S.cfg = Object.assign({}, DEFAULTS, c.restoreMs ? { restoreMs: c.restoreMs } : {});
  }

  function begin(payload) {
    if (S.run && !isDone(S.run)) cancel(S.run, S.env, 'superseded');
    S.run = createRun(payload, root.performance ? root.performance.now() : Date.now());
    S.lastToken = payload.token;
    S.log.push({ ev: 'begin', token: payload.token });
  }

  function poll() {
    try {
      const env = S.env, pub = env.value('url-init', 'data');
      if (pub && pub.phase === 'ready' && pub.payload && pub.payload.status === 'active' && pub.payload.token !== S.lastToken) {
        begin(pub.payload);
      } else if (pub && pub.phase === 'failed' && pub.hadOwnedParams === true && S.lastNotice !== pub.epoch) {
        S.lastNotice = pub.epoch;
        showNotice('This link could not be read, so the default view is shown.');
      } else if (pub && pub.phase === 'ready' && pub.payload && pub.payload.status === 'none' && pub.payload.hasOwnedParams
        && S.lastNotice !== pub.epoch) {
        S.lastNotice = pub.epoch;
        showNotice(noticeText(null, pub.payload.warnings) || 'This link did not contain a view that could be restored.');
      }
      const now = root.performance ? root.performance.now() : Date.now();
      if (S.run && !isDone(S.run)) step(S.run, env, now, S.cfg);
      syncTick(env, pub, now);
    } catch (err) {
      S.log.push({ ev: 'error', message: String(err && err.message || err) });
      if (S.run && !isDone(S.run)) {
        try { cancel(S.run, S.env, 'error'); } catch (err2) { /* last resort: keep the app usable */ }
        if (S.run && S.run.status === 'running') S.run.status = 'cancelled';
      }
    }
  }

  function viewIsSettled(env, pub, now) {
    const decided = !!pub && (pub.phase === 'ready' || pub.phase === 'failed');
    const restoring = !!S.run && !isDone(S.run);
    // a ready/active initialisation whose run has not started yet counts as "not decided"
    const starting = !!pub && pub.phase === 'ready' && !!pub.payload && pub.payload.status === 'active' && pub.payload.token !== S.lastToken;
    const meta = env.value('derived-model-metadata', 'data'), want = expectedKey(env);
    const key = meta && meta.key;
    const loaded = !!key && key[0] === want[0] && key[1] === want[1] && sameSet(key[2], want[2]);
    return { decided: decided && !starting, restoring, quiet: env.quiet(now, S.cfg.settleMs), loaded };
  }

  function requestSync(env, kind) {
    const id = env.uid();
    S.sync.reqId = id;
    env.set('url-sync-request', { data: { id, kind, snapshot: collectSnapshot(env, root.location.search) } });
    env.markWrite();
    return id;
  }

  function syncTick(env, pub, now) {
    const st = viewIsSettled(env, pub, now);
    if (S.share) shareTick(env, st, now);
    const decision = syncDecide(S.sync, collectSnapshot(env, ''), st, now, S.cfg.syncDebounceMs);
    if (decision === 'request' && !S.share) requestSync(env, 'live');
    const res = env.value('url-sync-result', 'data');
    if (res && res.id && res.id !== S.sync.lastResultId) {
      S.sync.lastResultId = res.id;
      if (res.kind === 'live') applyLive(res);
      else if (res.kind === 'share') finishShare(res);
    }
  }

  /** history.replaceState only: no new history entries, and Dash's Location is never told (so nothing reloads). */
  function applyLive(res) {
    if (res.status !== 'ok') { S.sync.lastSent = null; return; }              // not settled for the server: try again later
    if (res.search === root.location.search) return;
    root.history.replaceState(root.history.state, '', root.location.pathname + res.search + root.location.hash);
    S.sync.applied.push(res.search);
  }

  function showShare(text, link) {
    try {
      const dc = root.dash_clientside;
      dc.set_props('url-share-toast-text', { children: text });
      dc.set_props('url-share-link', { value: link || '', style: { display: link ? 'inline-block' : 'none', width: '300px', marginLeft: '8px' } });
      dc.set_props('url-share-toast', { style: Object.assign(noticeStyle(!!text), { left: '50%', transform: 'translateX(-50%)', background: '#e7f4e8', border: '1px solid #8fc796' }) });
      if (text && !link) root.setTimeout(() => { if (!S.share) showShare('', ''); }, 4000);
    } catch (err) { /* feedback is best effort */ }
  }

  function startShare() {
    if (S.share) return;
    S.share = { since: root.performance ? root.performance.now() : Date.now(), id: null };
    showShare('Preparing link…', '');
  }

  function shareTick(env, st, now) {
    const sh = S.share;
    if (!sh || sh.id) return;
    if (!st.decided || st.restoring || !st.loaded || !st.quiet) {                 // never copy an intermediate state
      if (now - sh.since > S.cfg.shareWaitMs) { S.share = null; showShare('The view is still loading. Try again in a moment.', ''); }
      return;
    }
    sh.id = requestSync(env, 'share');
  }

  function finishShare(res) {
    const sh = S.share;
    if (!sh || sh.id !== res.id) return;
    S.share = null;
    if (res.status !== 'ok') { showShare('Could not prepare the link. Try again once the view has finished loading.', ''); return; }
    const url = root.location.origin + root.location.pathname + res.search;
    S.lastShared = url;
    copyText(url).then(ok => { if (ok) showShare('Link copied to the clipboard.', ''); else showShare('Copy this link:', url); });
  }

  function legacyCopy(text) {
    try {
      const d = root.document, ta = d.createElement('textarea');
      ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
      d.body.appendChild(ta); ta.select();
      const ok = d.execCommand && d.execCommand('copy');
      d.body.removeChild(ta);
      return !!ok;
    } catch (err) { return false; }
  }
  function copyText(text) {
    try {
      if (root.navigator && root.navigator.clipboard && root.isSecureContext) {
        return root.navigator.clipboard.writeText(text).then(() => true, () => legacyCopy(text));
      }
    } catch (err) { /* fall through to the legacy path */ }
    return Promise.resolve(legacyCopy(text));
  }

  function installUserGate() {
    const d = root.document;
    const handler = e => {
      const run = S.run;
      if (!run || isDone(run) || !e.isTrusted) return;
      if (e.type === 'keydown' && !countsAsKey(e)) return;
      const kind = classifyTarget(e.target, d);
      if (!kind) return;
      cancel(run, S.env, kind === 'preset' ? 'preset_loaded' : 'user_changed');
    };
    ['pointerdown', 'keydown', 'input', 'change'].forEach(t => d.addEventListener(t, handler, true));
    d.addEventListener('click', e => {
      const tgt = e.target && e.target.closest ? e.target : null;
      if (!tgt) return;
      if (tgt.closest('#url-restore-notice-dismiss')) showNotice('');
      if (tgt.closest('#share-view-btn')) startShare();
    }, true);
  }

  function start() {
    if (S.timer || !root.document || !root.dash_clientside) return;
    installNetworkProbe();
    S.env = makeEnv();
    readConfig(S.env);
    installUserGate();
    S.timer = root.setInterval(poll, DEFAULTS.pollMs);
  }

  function debug() {
    const r = S.run;
    return { token: S.lastToken, log: S.log.slice(), net: Object.assign({}, net),
      sync: { applied: S.sync.applied.slice(), lastSent: S.sync.lastSent }, shared: S.lastShared || null,
      run: r ? { token: r.token, stage: r.stage, status: r.status, reason: r.reason, applied: r.applied, dropped: r.dropped,
        adjusted: r.adjusted, writes: r.writes, log: r.log.slice() } : null };
  }

  if (root && typeof root.document !== 'undefined') {
    installNetworkProbe();                                       // before the Dash renderer issues its first request
    const boot = () => { if (root.dash_clientside) start(); else root.setTimeout(boot, 50); };
    if (root.document.readyState === 'loading') root.document.addEventListener('DOMContentLoaded', boot); else boot();
  }

  return { DEFAULTS, IDENTITY, FILTER_ORDER, ADAPTERS, labelsToTick, clusterTicks, createRun, step, cancel, noticeText,
    classifyTarget, countsAsKey, sameSet, sameRange, collectSnapshot, syncDecide, canon, start, debug, net };
});
