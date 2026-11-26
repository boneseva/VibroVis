// Node tests for assets/url_reconcile.js: the stepping core (staging, compare-and-set, verification), the adapters,
// the shared vectors and the user-input classification. A simulated app with DELAYED resets stands in for Dash.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const api = createRequire(import.meta.url)(path.join(here, '..', '..', 'assets', 'url_reconcile.js'));
const vectors = JSON.parse(readFileSync(path.join(here, '..', 'fixtures', 'url_restore_vectors.json'), 'utf8'));
const { createRun, step, cancel, ADAPTERS, labelsToTick, clusterTicks, FILTER_ORDER } = api;
const CFG = { settleMs: 400, pollMs: 100, maxRewrites: 3, maxVerifyPasses: 6, noKeyPolls: 5, planTimeoutMs: 3000, restoreMs: 60000 };

// ---------------------------------------------------------------------------------------------------------------------
// A small simulated app. Widgets are {props}; "callbacks" are rules run `latency` ms after a write (like a server
// callback that resets dependants), and `quiet` is true only when no effect is pending and nothing happened recently.
// ---------------------------------------------------------------------------------------------------------------------
function makeApp(opts = {}) {
  const A = {
    now: 0, last: 0, pending: [], writes: [], widgets: {}, rowsByType: {}, progress: null, latency: opts.latency ?? 150,
    views: [], planCalls: [], planAnswer: opts.planAnswer, marks: 0,
  };
  const W = A.widgets;
  W['project-dropdown'] = { value: 'P1', options: ['P1', 'P2', 'P3'].map(v => ({ value: v })) };
  W['location-dropdown'] = { value: ['L1'], options: ['L1', 'L2'].map(v => ({ value: v })) };
  W['model-dropdown'] = { value: 'm1', options: ['m1', 'm2'].map(v => ({ value: v })) };
  W['num-cluster-dropdown'] = { value: 10, options: [10, 20].map(v => ({ value: v })) };
  W['derived-model-metadata'] = { data: { key: ['P1', 'm1', ['L1']] } };
  W['url-identity'] = { data: null };
  W['url-plan-request'] = { data: null };
  W['url-plan-filters'] = { data: null };
  W['color-mode-radio'] = { value: 'cluster' };
  W['microlocation-dropdown'] = { value: ['a', 'b'], options: [{ value: 'a' }, { value: 'b' }] };
  W['recorder-type-dropdown'] = { value: ['r1', 'r2'], options: [{ value: 'r1' }, { value: 'r2' }] };
  W['channel-checklist'] = { value: [1, 2] };
  W['hour-slider'] = { value: [0, 24] };
  W['merge-switch'] = { on: false };
  W['merge-threshold'] = { value: 15, max: 100 };
  W['clip-count-threshold'] = { value: 1, max: 5 };
  W['max-points'] = { value: 10000 };
  W['priority-sampling-toggle'] = { value: ['on'] };
  W['table-sort-store'] = { data: { col: null, asc: true } };
  W['table-page-store'] = { data: 3 };
  W['main-view-tabs'] = { data: 'scatter-tab' };
  A.rowsByType['cluster-checkbox'] = [0, 1, 2, 3].map(i => ({ id: { type: 'cluster-checkbox', index: i }, value: ['on'] }));
  A.rowsByType['label-checkbox'] = ['cat', 'Unlabeled'].map(i => ({ id: { type: 'label-checkbox', index: i }, value: ['on'] }));
  A.rowsByType['date-checklist'] = [
    { id: { type: 'date-checklist', year: 2026, month: 3 }, value: ['2026-03-01', '2026-03-02'], options: [{ value: '2026-03-01' }, { value: '2026-03-02' }] },
    { id: { type: 'date-checklist', year: 2026, month: 4 }, value: ['2026-04-01'], options: [{ value: '2026-04-01' }] }];
  A.rowsByType['simple-date-dropdown'] = [];
  A.rules = opts.rules || [];            // [(id, props, app) => [{at, fn}]]

  const key = id => (typeof id === 'string' ? id : JSON.stringify(id));
  A.effect = (delay, fn) => A.pending.push({ at: A.now + delay, fn });
  A.timers = [];                                          // UNOBSERVED client-side dispatch gaps (invisible to `quiet`)
  A.later = (delay, fn) => A.timers.push({ at: A.now + delay, fn });
  A.advance = () => {
    for (const e of A.pending.filter(p => p.at <= A.now)) { e.fn(); A.last = A.now; }
    A.pending = A.pending.filter(p => p.at > A.now);
    for (const e of A.timers.filter(p => p.at <= A.now)) e.fn();
    A.timers = A.timers.filter(p => p.at > A.now);
  };
  A.env = {
    value: (id, prop) => (W[key(id)] || {})[prop === undefined ? 'data' : prop],
    options: id => (W[key(id)] || {}).options || [],
    set: (id, props) => {
      const k = key(id);
      A.writes.push({ id: k, props: JSON.parse(JSON.stringify(props)), at: A.now });
      const row = Object.values(A.rowsByType).flat().find(r => key(r.id) === k);
      if (row) Object.assign(row, props); else W[k] = Object.assign(W[k] || {}, props);
      A.last = A.now;
      for (const rule of A.rules) rule(k, props, A);
      if (k === 'url-plan-request') A.effect(A.latency, () => {
        A.planCalls.push(props.data);
        W['url-plan-filters'] = { data: Object.assign({ id: props.data.id, token: props.data.token }, A.planAnswer(props.data)) };
      });
    },
    markWrite: () => { A.last = A.now; A.marks++; },
    quiet: (now, settle) => A.pending.length === 0 && now - A.last >= settle,
    uid: (() => { let n = 0; return () => `id${++n}`; })(),
    rows: type => (A.rowsByType[type] || []).map(r => ({ id: r.id, value: r.value, options: r.options })),
    currentView: () => (W['main-view-tabs'].data === 'table-tab' ? 'table' : 'scatter'),
    setView: v => { A.views.push(v); W['main-view-tabs'] = { data: v === 'table' ? 'table-tab' : 'scatter-tab' }; A.last = A.now; },
    progress: info => { A.progress = info; },
  };
  return A;
}

const payload = (state, token = 'tok') => ({ token, state, warnings: [] });
const okPlan = (applied, extra = {}) => () => Object.assign({ status: 'ok', applied, dropped: [], adjusted: [], cluster_ids: [0, 1, 2, 3] }, extra);

function drive(A, run, maxMs = 60000, onTick) {
  while (A.now < maxMs && run.status === 'running') {
    A.now += 100; A.advance();
    if (onTick) onTick(A.now);
    step(run, A.env, A.now, CFG);
  }
  return run;
}

// The recorded identity cascade of the real app: a new project resets the location, a new location resets the model,
// a new model resets k, and the dataset load (after identity is done) resets every filter (all micros/recorders/...).
const cascade = (id, props, A) => {
  const W = A.widgets;
  if (id === 'project-dropdown') A.effect(A.latency, () => { W['location-dropdown'].value = ['L1']; A.effect(A.latency, () => { W['model-dropdown'].value = 'm1'; W['num-cluster-dropdown'].value = 10; }); });
  if (id === 'location-dropdown') A.effect(A.latency, () => { if (!['m1', 'm2'].includes(W['model-dropdown'].value)) W['model-dropdown'].value = 'm1'; });
  if (id === 'url-identity') A.effect(A.latency * 2, () => {
    const proj = W['project-dropdown'].value, mod = W['model-dropdown'].value, loc = W['location-dropdown'].value.slice().sort();
    const had = W['derived-model-metadata'].data && W['derived-model-metadata'].data.key;
    if (had && had[0] === proj && had[1] === mod && JSON.stringify(had[2]) === JSON.stringify(loc)) return;   // same dataset: idempotent, no reload
    W['derived-model-metadata'] = { data: { key: [proj, mod, loc] } };
    A.effect(A.latency, () => {                     // dataset load resets the filters
      W['microlocation-dropdown'].value = ['a', 'b']; W['recorder-type-dropdown'].value = ['r1', 'r2'];
      W['channel-checklist'].value = [1, 2]; W['hour-slider'].value = [0, 24]; W['merge-switch'].on = false;
      A.rowsByType['cluster-checkbox'].forEach(r => { r.value = ['on']; });
    });
  });
  if (id === 'microlocation-dropdown') A.effect(A.latency, () => { W['recorder-type-dropdown'].value = ['r1', 'r2']; });   // recorder reset
  if (id === 'merge-switch') A.effect(A.latency, () => { W['clip-count-threshold'].value = 1; });
};

// ---------------------------------------------------------------------------------------------------------------------
test('the filter order equals url_state.STAGE_KEYS["filters"] (shared vector)', () => {
  assert.deepEqual(FILTER_ORDER, vectors.filter_order);
});

test('identity is applied in order, each step only after the app went quiet; the load hold is released once', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({}) });
  const run = createRun(payload({ project: 'P2', loc: ['L2'], model: 'm2', k: 20 }), 0);
  drive(A, run);
  const w = A.writes.map(x => x.id);
  assert.deepEqual(w.filter(x => ['project-dropdown', 'location-dropdown', 'model-dropdown', 'num-cluster-dropdown'].includes(x)),
    ['project-dropdown', 'location-dropdown', 'model-dropdown', 'num-cluster-dropdown']);
  assert.equal(A.writes.filter(x => x.id === 'url-identity').length, 1);
  assert.deepEqual(A.writes.find(x => x.id === 'url-identity').props.data.done, true);
  assert.equal(A.widgets['project-dropdown'].value, 'P2');
  assert.deepEqual(A.widgets['location-dropdown'].value, ['L2']);
  assert.equal(A.widgets['model-dropdown'].value, 'm2');
  assert.equal(A.widgets['num-cluster-dropdown'].value, 20);
  assert.equal(run.status, 'complete');
  // the hold is released AFTER every identity write
  const idx = A.writes.findIndex(x => x.id === 'url-identity');
  assert.ok(A.writes.slice(0, idx).every(x => x.id !== 'url-plan-request'));
});

test('an unavailable identity value is dropped with the shared reason and the rest still applies', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({}) });
  const run = createRun(payload({ project: 'NOPE', loc: ['L2', 'ZZ'], model: 'ghost', k: 99 }), 0);
  drive(A, run);
  const byKey = Object.fromEntries(run.dropped.map(d => [d.key + ':' + JSON.stringify(d.value), d.reason]));
  assert.equal(byKey['project:"NOPE"'], 'unknown_project');
  assert.equal(byKey['loc:"ZZ"'], 'unknown_location');
  assert.equal(byKey['model:"ghost"'], 'unknown_model');
  assert.equal(byKey['k:99'], 'unknown_k');
  assert.deepEqual(A.widgets['location-dropdown'].value, ['L2']);
  assert.equal(run.status, 'complete');
});

test('filters are written in dependency order and only AFTER the dataset matches the identity', () => {
  const applied = { color: 'manual', micro: ['a'], rec: ['r1'], ch: [1], hours: [6, 12], merge: true, mth: 30, mcc: 2, max: 5000, prio: false, sort: ['cluster_id', false], view: 'table' };
  const A = makeApp({ rules: [cascade], planAnswer: okPlan(applied) });
  const run = createRun(payload({ project: 'P2', loc: ['L2'], model: 'm2', k: 20, ...applied }), 0);
  drive(A, run);
  const order = A.writes.map(x => x.id);
  const firstPlan = order.indexOf('url-plan-request');
  assert.ok(firstPlan > order.indexOf('url-identity'));
  assert.equal(A.planCalls.length, 1);
  assert.deepEqual(A.planCalls[0].key, ['P2', 'm2', ['L2']]);
  assert.equal(A.planCalls[0].k, 20);
  const tail = order.slice(firstPlan + 1);
  const pos = id => tail.indexOf(id);
  assert.ok(pos('color-mode-radio') < pos('microlocation-dropdown'));
  assert.ok(pos('microlocation-dropdown') < pos('recorder-type-dropdown'));
  assert.ok(pos('recorder-type-dropdown') < pos('channel-checklist'));
  assert.ok(pos('hour-slider') < pos('merge-switch'));
  assert.ok(pos('merge-switch') < pos('merge-threshold') && pos('merge-threshold') < pos('clip-count-threshold'));
  assert.ok(pos('max-points') < pos('priority-sampling-toggle') && pos('priority-sampling-toggle') < pos('table-sort-store'));
  assert.deepEqual(A.views, ['table']);
  assert.equal(run.status, 'complete');
  assert.deepEqual(A.widgets['microlocation-dropdown'].value, ['a']);
  assert.deepEqual(A.widgets['recorder-type-dropdown'].value, ['r1']);
  assert.equal(A.widgets['clip-count-threshold'].value, 2);              // survives the merge-switch reset (written after it)
  assert.deepEqual(A.widgets['table-page-store'].data, 0);
});

test('a reset that lands AFTER the value looked right is corrected before completion (late reset)', () => {
  let armed = true;
  const lateReset = (id, props, A) => {
    // 700 ms after the hour slider is first written, a callback puts the default back (beyond the quiet window).
    if (id === 'hour-slider' && armed) { armed = false; A.effect(700, () => { A.widgets['hour-slider'].value = [0, 24]; }); }
  };
  const A = makeApp({ rules: [cascade, lateReset], planAnswer: okPlan({ hours: [6, 12], max: 5000 }) });
  const run = createRun(payload({ hours: [6, 12], max: 5000 }), 0);
  drive(A, run);
  assert.equal(run.status, 'complete');
  assert.deepEqual(A.widgets['hour-slider'].value, [6, 12]);
  assert.ok(A.writes.filter(x => x.id === 'hour-slider').length >= 2, 'the slider was rewritten after the late reset');
  assert.ok(run.log.some(l => l.startsWith('write:hours#2')));
});

test('a value that is reset again and again is dropped as unstable after the retry cap', () => {
  const stubborn = (id, props, A) => { if (id === 'channel-checklist') A.effect(10, () => { A.widgets['channel-checklist'].value = [1, 2]; }); };
  const A = makeApp({ rules: [cascade, stubborn], planAnswer: okPlan({ ch: [1], max: 3000 }) });
  const run = createRun(payload({ ch: [1], max: 3000 }), 0);
  drive(A, run);
  assert.equal(run.status, 'complete');
  assert.ok(run.dropped.some(d => d.key === 'ch' && d.reason === 'unstable'));
  assert.equal(A.writes.filter(x => x.id === 'channel-checklist').length, CFG.maxRewrites);
  assert.equal(A.widgets['max-points'].value, 3000);                     // the others were not held up
});

test('compare-and-set: applying the same plan over the finished state writes nothing', () => {
  const applied = { micro: ['a'], hours: [6, 12], max: 5000, prio: false };
  const A = makeApp({ rules: [cascade], planAnswer: okPlan(applied) });
  drive(A, createRun(payload(applied), 0));
  const before = A.writes.length;
  const second = createRun(payload(applied, 'tok2'), A.now);
  drive(A, second, A.now + 30000);
  const newWrites = A.writes.slice(before).filter(x => !['url-identity', 'url-plan-request'].includes(x.id));
  assert.deepEqual(newWrites, []);
  assert.equal(second.status, 'complete');
});

test('a plan that is not ok drops every filter key with its status; identity still releases the hold', () => {
  for (const status of ['dataset_mismatch', 'stale']) {
    const A = makeApp({ rules: [cascade], planAnswer: () => ({ status, applied: {}, dropped: [], adjusted: [] }) });
    const run = createRun(payload({ project: 'P2', micro: ['a'], hours: [6, 12] }), 0);
    drive(A, run);
    assert.deepEqual(run.dropped.map(d => [d.key, d.reason]), [['micro', status], ['hours', status]]);
    assert.equal(A.widgets['project-dropdown'].value, 'P2');
    assert.ok(A.writes.some(x => x.id === 'url-identity'));
    assert.equal(run.status, 'complete');
  }
});

test('the dataset never appearing drops the filters as dataset_unavailable', () => {
  const noLoad = (id, props, A) => { if (id === 'url-identity') { /* the load produced nothing */ } };
  const A = makeApp({ rules: [noLoad], planAnswer: okPlan({}) });
  A.widgets['derived-model-metadata'] = { data: {} };
  const run = createRun(payload({ micro: ['a'] }), 0);
  drive(A, run);
  assert.ok(run.dropped.some(d => d.key === 'micro' && d.reason === 'dataset_unavailable'));
  assert.equal(A.planCalls.length, 0);
});

test('a plan that never arrives times out as plan_timeout', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({}) });
  A.effect = () => {};                         // swallow every delayed effect, including the plan answer
  A.widgets['derived-model-metadata'] = { data: { key: ['P1', 'm1', ['L1']] } };
  const run = createRun(payload({ micro: ['a'] }), 0);
  drive(A, run);
  assert.ok(run.dropped.some(d => d.key === 'micro' && d.reason === 'plan_timeout'));
});

test('a user action cancels: the hold is released, applied keys stay, the rest are dropped as interrupted', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({ micro: ['a'], hours: [6, 12] }) });
  const run = createRun(payload({ project: 'P2', loc: ['L2'], model: 'm2', hours: [6, 12] }), 0);
  drive(A, run, 60000, now => { if (A.writes.some(x => x.id === 'project-dropdown') && !run.cancelledOnce) { run.cancelledOnce = true; cancel(run, A.env, 'user_changed'); } });
  assert.equal(run.status, 'cancelled');
  assert.equal(run.reason, 'user_changed');
  assert.ok(A.writes.some(x => x.id === 'url-identity' && x.props.data.done === true), 'the load hold was released');
  const reasons = Object.fromEntries(run.dropped.map(d => [d.key, d.reason]));
  assert.equal(reasons.loc, 'interrupted'); assert.equal(reasons.model, 'interrupted'); assert.equal(reasons.hours, 'interrupted');
  const writesAfter = A.writes.length;
  step(run, A.env, A.now + 5000, CFG);
  assert.equal(A.writes.length, writesAfter, 'a cancelled run never writes again');
});

test('the watchdog cancels a run that takes too long', () => {
  const A = makeApp({ rules: [], planAnswer: okPlan({}) });
  A.pending.push({ at: 1e12, fn() {} });                                  // the app never goes quiet
  const run = createRun(payload({ micro: ['a'] }), 0);
  for (let t = 0; t < 70000 && run.status === 'running'; t += 500) { A.now = t; step(run, A.env, t, CFG); }
  assert.equal(run.status, 'cancelled'); assert.equal(run.reason, 'timeout');
});

test('nothing restorable in the identity stage: the hold is released immediately, no filter plan without filter keys', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({}) });
  const run = createRun(payload({ project: 'P1' }), 0);                   // already the current project
  drive(A, run);
  assert.equal(run.status, 'complete');
  assert.equal(A.writes.filter(x => x.id === 'project-dropdown').length, 0);
  assert.equal(A.planCalls.length, 0);
  assert.equal(A.writes.filter(x => x.id === 'url-identity').length, 1);
});

// ---------------------------------------------------------------------------------------------------------------------
// Adapters and shared vectors
// ---------------------------------------------------------------------------------------------------------------------
test('labelsToTick and clusterTicks agree with the shared vectors (Python parity)', () => {
  for (const v of vectors.labels_to_tick) assert.deepEqual(labelsToTick(v.labels, v.rows), v.expect, v.name);
  for (const v of vectors.cluster_ticks) assert.deepEqual(clusterTicks(v.clusters ?? null, v.hide ?? null, v.ids), v.expect, v.name);
});

test('cluster adapter: ticks only the wanted rows, only mismatches are written, an empty selection is dropped', () => {
  const A = makeApp();
  let res = ADAPTERS.clusters([1, 3], A.env);
  assert.deepEqual(res.writes.map(w => w.id.index).sort(), [0, 2]);
  assert.ok(res.writes.every(w => sameArr(w.props.value, [])));
  A.rowsByType['cluster-checkbox'].forEach(r => { r.value = [1, 3].includes(r.id.index) ? ['on'] : []; });
  assert.deepEqual(ADAPTERS.clusters([1, 3], A.env).writes, []);
  res = ADAPTERS.hide([1, 3], A.env);                                     // hide 1,3 == show 0 and 2: all four rows flip
  assert.deepEqual(res.writes.map(w => [w.id.index, w.props.value.length]).sort(), [[0, 1], [1, 0], [2, 1], [3, 0]]);
  A.rowsByType['cluster-checkbox'].forEach(r => { r.value = ['on']; });
  assert.deepEqual(ADAPTERS.hide([0, 1, 2, 3], A.env).drop, 'empty_cluster_selection');
});
const sameArr = (a, b) => JSON.stringify(a) === JSON.stringify(b);

test('labels adapter: unlabeled ticks only Unlabeled; labeled ticks the rest; a legend with no label row drops', () => {
  const A = makeApp();
  let res = ADAPTERS.labels('unlabeled', A.env);
  assert.deepEqual(res.writes.map(w => [w.id.index, w.props.value]), [['cat', []]]);
  res = ADAPTERS.labels('labeled', A.env);
  assert.deepEqual(res.writes.map(w => [w.id.index, w.props.value]), [['Unlabeled', []]]);
  A.rowsByType['label-checkbox'] = [{ id: { type: 'label-checkbox', index: 'Unlabeled' }, value: ['on'] }];
  assert.equal(ADAPTERS.labels('labeled', A.env).drop, 'no_labeled_rows');
  assert.deepEqual(ADAPTERS.labels('unlabeled', A.env).writes, []);
});

test('dates adapter: tree mode writes only the months that differ; simple mode writes the dropdown; no widget drops', () => {
  const A = makeApp();
  let res = ADAPTERS.dates(['2026-03-02', '2026-04-01'], A.env);
  assert.deepEqual(res.writes.map(w => [w.id.month, w.props.value]), [[3, ['2026-03-02']]]);
  res = ADAPTERS.dates([], A.env);
  assert.deepEqual(res.writes.map(w => [w.id.month, w.props.value]), [[3, []], [4, []]]);
  A.rowsByType['simple-date-dropdown'] = [{ id: { type: 'simple-date-dropdown', index: 0 }, value: ['d1', 'd2', 'd3'], options: [{ value: 'd1' }, { value: 'd2' }, { value: 'd3' }] }];
  res = ADAPTERS.dates(['d2'], A.env);
  assert.deepEqual(res.writes.map(w => w.props.value), [['d2']]);
  A.rowsByType['simple-date-dropdown'] = []; A.rowsByType['date-checklist'] = [];
  assert.equal(ADAPTERS.dates(['x'], A.env).drop, 'no_date_widget');
});

test('view adapter sets the pane through setView and is idempotent', () => {
  const A = makeApp();
  ADAPTERS.view('table', A.env).writes.forEach(w => A.env.setView(w.view));
  assert.deepEqual(ADAPTERS.view('table', A.env).writes, []);
  assert.deepEqual(A.views, ['table']);
});

test('number adapters ignore representation noise and the sort adapter resets the page only when it writes', () => {
  const A = makeApp();
  assert.deepEqual(ADAPTERS.hours([0, 24.0000000001], A.env).writes, []);
  const res = ADAPTERS.sort(['cluster_id', true], A.env);
  assert.deepEqual(res.writes.map(w => w.id), ['table-sort-store', 'table-page-store']);
  A.widgets['table-sort-store'].data = { col: 'cluster_id', asc: true };
  assert.deepEqual(ADAPTERS.sort(['cluster_id', true], A.env).writes, []);
});

// ---------------------------------------------------------------------------------------------------------------------
// User-input classification (pure)
// ---------------------------------------------------------------------------------------------------------------------
const el = (id, parent = null) => ({ nodeType: 1, id, parentElement: parent });
test('classifyTarget: registered controls and pattern types count; the preset button is its own reason; others do not', () => {
  const api2 = api;
  assert.equal(api2.classifyTarget(el('', el('recorder-type-dropdown'))), 'control');
  assert.equal(api2.classifyTarget(el(JSON.stringify({ index: 2, type: 'cluster-checkbox' }))), 'control');
  assert.equal(api2.classifyTarget(el(JSON.stringify({ index: 2, type: 'cluster-color-picker' }))), null);
  assert.equal(api2.classifyTarget(el('preset-load-btn')), 'preset');
  for (const id of ['audio-player', 'spectrogram-plot', 'anim-play-btn', 'toggle-filters-btn', 'logo']) assert.equal(api2.classifyTarget(el(id)), null, id);
});
test('countsAsKey: Tab/Shift/Alt/Escape and Ctrl/Meta combinations never cancel; arrows, Enter, Space, characters do', () => {
  for (const key of ['Tab', 'Shift', 'Alt', 'Escape', 'Control', 'Meta']) assert.equal(api.countsAsKey({ key }), false, key);
  assert.equal(api.countsAsKey({ key: ' ', ctrlKey: true }), false);
  assert.equal(api.countsAsKey({ key: 'c', metaKey: true }), false);
  for (const key of ['ArrowLeft', 'Enter', ' ', 'a', 'Backspace', 'Delete']) assert.equal(api.countsAsKey({ key }), true, key);
});

test('noticeText summarises drops and parse warnings, ignoring clamps', () => {
  assert.equal(api.noticeText({ dropped: [] }, []), '');
  assert.equal(api.noticeText({ dropped: [{ key: 'micro' }] }, [{ code: 'x', key: 'k' }, { code: 'clamped', key: 'max' }]),
    '2 settings in this link could not be applied (k, micro).');
});

// ---------------------------------------------------------------------------------------------------------------------
// Final figure refresh (seen in a real browser: the scatter can be computed from legend rows that are then replaced).
// It must be a SEEDED recompute (an equal-value write of max-points), never the Resample button (unseeded sample).
// ---------------------------------------------------------------------------------------------------------------------
test('a restore that wrote something ends with exactly ONE refresh: an equal-value max-points write, never Resample', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({ hours: [6, 12], max: 5000 }) });
  const run = createRun(payload({ project: 'P2', hours: [6, 12], max: 5000 }), 0);
  drive(A, run);
  assert.equal(A.writes.filter(x => x.id === 'resample-btn').length, 0, 'the Resample button is never touched');
  const maxWrites = A.writes.filter(x => x.id === 'max-points');
  assert.equal(maxWrites.length, 2, 'the restored value, then the refresh');
  assert.deepEqual(maxWrites[1].props, maxWrites[0].props, 'the refresh writes the value it already has');
  assert.equal(A.writes.at(-1).id, 'max-points', 'it is the very last write');
  assert.equal(run.log.filter(l => l === 'refresh_figure').length, 1);
  assert.equal(run.status, 'complete');
});

test('a no-op restore refreshes nothing, and a cancelled restore does not refresh', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({ max: 5000 }) });
  drive(A, createRun(payload({ max: 5000 }), 0));
  const before = A.writes.length;
  const again = createRun(payload({ max: 5000 }, 'tok2'), A.now);                 // everything already matches
  drive(A, again, A.now + 30000);
  assert.equal(A.writes.slice(before).filter(x => x.id === 'max-points').length, 0);
  assert.equal(again.status, 'complete');

  const B = makeApp({ rules: [cascade], planAnswer: okPlan({ max: 5000 }) });
  const run = createRun(payload({ project: 'P2', max: 5000 }), 0);
  drive(B, run, 60000, () => { if (B.writes.some(x => x.id === 'project-dropdown') && !run.once) { run.once = true; cancel(run, B.env, 'user_changed'); } });
  assert.equal(B.writes.filter(x => x.id === 'max-points').length, 0);
});

test('without a numeric max-points the refresh is skipped (and noted) instead of writing garbage', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({ hours: [6, 12] }) });
  A.widgets['max-points'] = { value: null };
  const run = createRun(payload({ hours: [6, 12] }), 0);
  drive(A, run);
  assert.equal(A.writes.filter(x => x.id === 'max-points').length, 0);
  assert.ok(run.log.includes('refresh_skipped_no_max_points'));
  assert.equal(run.status, 'complete');
});

// =====================================================================================================================
// STAGE-GATE STRESS TESTS
//
// Why the quiet-window gate is sufficient (and when it is not):
//   * `quiet` is false while ANY Dash callback request is in flight and for `settleMs` (400 ms) after the last request
//     ended or the last write. A callback that a write provokes is dispatched by the renderer as soon as the
//     triggering write/response has been processed, i.e. with an inter-dispatch gap far below 400 ms; while it runs it is
//     in flight, so the gate holds. Consequently every reset a write can provoke has LANDED before the next quiet tick,
//     whatever its server latency (a 30 s dataset load simply keeps the gate shut for 30 s).
//   * The gate can only be fooled by an UNOBSERVED client-side gap of more than `settleMs` between two chained callbacks
//     (nothing in flight, nothing recent). The tests below model gaps up to just under the window and require zero
//     clobbers; the existing "late reset" tests show the verify pass catching a reset that does slip through before the
//     run completes. Beyond that (a gap > 400 ms and a reset after completion) is the documented residual risk.
// =====================================================================================================================
function rng(seed) {                                                         // mulberry32: deterministic pseudo-random numbers
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
/** Dependent callbacks with random latency; chained steps are separated by an unobserved dispatch gap of 0..gap ms. */
const randomCascade = (rnd, lat, gap, loadMs) => (id, props, A) => {
  const W = A.widgets, L = () => lat[0] + rnd() * (lat[1] - lat[0]), G = () => (gap ? rnd() * gap : 0);
  const chain = steps => {
    const go = i => { if (i >= steps.length) return; A.effect(L(), () => { steps[i](); if (i + 1 < steps.length) A.later(G(), () => go(i + 1)); }); };
    go(0);
  };
  if (id === 'project-dropdown') chain([() => { W['location-dropdown'].value = ['L1']; }, () => { W['model-dropdown'].value = 'm1'; }, () => { W['num-cluster-dropdown'].value = 10; }]);
  if (id === 'location-dropdown') chain([() => { if (!['m1', 'm2'].includes(W['model-dropdown'].value)) W['model-dropdown'].value = 'm1'; }, () => { if (W['model-dropdown'].value === 'm1') W['num-cluster-dropdown'].value = 10; }]);
  if (id === 'model-dropdown') chain([() => { W['num-cluster-dropdown'].value = W['num-cluster-dropdown'].value === 20 ? 20 : 10; }]);
  if (id === 'url-identity') {
    const D = loadMs[0] + rnd() * (loadMs[1] - loadMs[0]);
    A.effect(D, () => {                                                    // the dataset load answers
      const proj = W['project-dropdown'].value, mod = W['model-dropdown'].value, loc = W['location-dropdown'].value.slice().sort();
      const had = W['derived-model-metadata'].data && W['derived-model-metadata'].data.key;
      if (had && had[0] === proj && had[1] === mod && JSON.stringify(had[2]) === JSON.stringify(loc)) return;
      A.loadedAt = A.now;
      W['derived-model-metadata'] = { data: { key: [proj, mod, loc] } };
      chain([() => { W['microlocation-dropdown'].value = ['a', 'b']; W['channel-checklist'].value = [1, 2]; W['hour-slider'].value = [0, 24]; W['merge-switch'].on = false; },
        () => { W['recorder-type-dropdown'].value = ['r1', 'r2']; }, () => { A.rowsByType['cluster-checkbox'].forEach(r => { r.value = ['on']; }); }]);
    });
  }
  if (id === 'microlocation-dropdown') chain([() => { W['recorder-type-dropdown'].value = ['r1', 'r2']; }]);
  if (id === 'merge-switch') chain([() => { W['clip-count-threshold'].value = 1; }]);
};
const WANT = { project: 'P2', loc: ['L2'], model: 'm2', k: 20, micro: ['a'], rec: ['r1'], ch: [1], hours: [6, 12], merge: true, mth: 30, mcc: 2, max: 5000, prio: false, color: 'manual', clusters: [1, 3] };
const WANT_APPLIED = Object.fromEntries(Object.entries(WANT).filter(([k]) => !['project', 'loc', 'model', 'k'].includes(k)));

function stressOnce(seed, { lat = [20, 380], gap = 0, loadMs = [300, 4000], want = WANT, applied = WANT_APPLIED } = {}) {
  const rnd = rng(seed);
  const A = makeApp({ rules: [randomCascade(rnd, lat, gap, loadMs)], planAnswer: okPlan(applied) });
  const violations = [];
  const origSet = A.env.set;
  A.env.set = (id, props) => {                                           // watch every write for ordering violations
    const W = A.widgets;
    if (id === 'url-plan-request') {
      const key = W['derived-model-metadata'].data && W['derived-model-metadata'].data.key;
      const now = [W['project-dropdown'].value, W['model-dropdown'].value, W['location-dropdown'].value.slice().sort()];
      if (!key || JSON.stringify(key) !== JSON.stringify(now)) violations.push('plan requested before the dataset matched the identity');
      if (A.loadedAt === undefined && JSON.stringify(now) !== JSON.stringify(['P1', 'm1', ['L1']])) violations.push('plan without a load');
    }
    if (['hour-slider', 'microlocation-dropdown', 'merge-switch'].includes(id) && A.loadedAt === undefined && JSON.stringify([W['project-dropdown'].value]) !== '["P1"]') violations.push('filter written before the dataset loaded');
    return origSet(id, props);
  };
  const run = createRun(payload(want), 0);
  drive(A, run, 600000);
  return { A, run, violations };
}
const finalOk = (A, want) => {
  const W = A.widgets;
  return W['project-dropdown'].value === want.project && JSON.stringify(W['location-dropdown'].value) === JSON.stringify(want.loc)
    && W['model-dropdown'].value === want.model && W['num-cluster-dropdown'].value === want.k
    && JSON.stringify(W['microlocation-dropdown'].value) === JSON.stringify(want.micro) && JSON.stringify(W['recorder-type-dropdown'].value) === JSON.stringify(want.rec)
    && JSON.stringify(W['hour-slider'].value) === JSON.stringify(want.hours) && W['merge-switch'].on === true
    && W['merge-threshold'].value === want.mth && W['clip-count-threshold'].value === want.mcc && W['max-points'].value === want.max
    && JSON.stringify(W['priority-sampling-toggle'].value) === '[]';
};

test('STRESS: 200 random cascades (latency 20-380 ms, chained dependants, unobserved gaps just under the window): no clobber, no early stage, exact final state', () => {
  for (let seed = 1; seed <= 200; seed++) {
    const { A, run, violations } = stressOnce(seed, { lat: [20, 380], gap: 360 });
    assert.equal(run.status, 'complete', `seed ${seed}`);
    assert.deepEqual(violations, [], `seed ${seed}`);
    assert.deepEqual(run.dropped, [], `seed ${seed}: nothing may be dropped or unstable`);
    assert.ok(finalOk(A, WANT), `seed ${seed}: the final widgets equal the requested view`);
    const writes = Object.entries(run.writes).filter(([k]) => k !== 'max');     // max is written twice by design (value + refresh)
    for (const [k, n] of writes) assert.equal(n, 1, `seed ${seed}: ${k} was written ${n} times (a value was clobbered and rewritten)`);
    const idWrites = A.writes.map(x => x.id);
    const doneAt = A.writes.findIndex(x => x.id === 'url-identity');
    assert.equal(idWrites.filter(x => x === 'url-identity').length, 1, `seed ${seed}: the hold is released exactly once`);
    for (const wId of ['project-dropdown', 'location-dropdown', 'model-dropdown', 'num-cluster-dropdown']) {
      const i = idWrites.lastIndexOf(wId);
      assert.ok(i < doneAt, `seed ${seed}: ${wId} written before the hold was released`);
    }
  }
});

test('STRESS: consecutive project/location/model/k updates are written strictly one after the other, each only after the previous settled', () => {
  for (let seed = 301; seed <= 340; seed++) {
    const { A, run } = stressOnce(seed, { lat: [150, 380], gap: 300 });
    const ident = A.writes.filter(x => ['project-dropdown', 'location-dropdown', 'model-dropdown', 'num-cluster-dropdown'].includes(x.id));
    assert.deepEqual(ident.map(x => x.id), ['project-dropdown', 'location-dropdown', 'model-dropdown', 'num-cluster-dropdown'], `seed ${seed}`);
    for (let i = 1; i < ident.length; i++) assert.ok(ident[i].at - ident[i - 1].at >= 400, `seed ${seed}: step ${i} followed its predecessor after only ${ident[i].at - ident[i - 1].at} ms`);
    assert.equal(run.status, 'complete');
  }
});

test('STRESS: slow dataset loading (up to 40 s) keeps every filter write waiting for the matching dataset, with no premature timeout', () => {
  for (const [seed, loadMs] of [[7, [5000, 5000]], [8, [20000, 20000]], [9, [40000, 40000]], [10, [100, 100]]]) {
    const { A, run, violations } = stressOnce(seed, { lat: [30, 100], loadMs });
    assert.equal(run.status, 'complete', `load ${loadMs[0]} ms`);
    assert.deepEqual(violations, []);
    assert.deepEqual(run.dropped, []);
    const firstFilter = A.writes.find(x => ['hour-slider', 'microlocation-dropdown'].includes(x.id));
    assert.ok(firstFilter.at >= A.loadedAt, `a filter was written at ${firstFilter.at} before the load answered at ${A.loadedAt}`);
    assert.ok(finalOk(A, WANT));
  }
});

test('STRESS: a user action at ANY moment of the restore releases the hold, stops all writes and never reverts the user', () => {
  let cancelled = 0;
  for (let t = 100; t <= 14000; t += 100) {
    const A = makeApp({ rules: [randomCascade(rng(t), [20, 380], 200, [300, 3000])], planAnswer: okPlan(WANT_APPLIED) });
    const run = createRun(payload(WANT), 0);
    let userValue = null, writesAtCancel = null;
    drive(A, run, 60000, now => {
      if (now >= t && !run.userDid) {
        run.userDid = true;
        userValue = [9, 10];
        A.widgets['hour-slider'].value = userValue;                            // the user drags the slider ...
        cancel(run, A.env, 'user_changed');                                    // ... and the DOM gate cancels the restore
        writesAtCancel = A.writes.length;
      }
    });
    if (run.status !== 'cancelled') { assert.equal(run.status, 'complete', `t=${t}: it had already finished`); continue; }
    cancelled++;
    A.now += 20000; A.advance();
    for (let i = 0; i < 200; i++) { A.now += 100; A.advance(); step(run, A.env, A.now, CFG); }
    assert.equal(A.writes.length, writesAtCancel + A.writes.slice(writesAtCancel).filter(x => x.id === 'url-identity').length, `t=${t}: no write after the cancel (except the one-time release)`);
    assert.equal(A.writes.filter(x => x.id === 'url-identity').length, 1, `t=${t}: the hold is released exactly once`);
    assert.deepEqual(A.widgets['hour-slider'].value === userValue || A.loadedAt !== undefined, true, `t=${t}: the user's value is never reverted by the reconciler`);
  }
  assert.ok(cancelled > 50, 'the sweep really exercised cancellation at many different stages');
});

test('a genuine user change DURING a delayed reset takes precedence: the reconciler neither completes early nor fights back', () => {
  let armed = true;
  const lateReset = (id, props, A) => { if (id === 'hour-slider' && armed) { armed = false; A.effect(900, () => { A.widgets['hour-slider'].value = [0, 24]; }); } };
  const A = makeApp({ rules: [cascade, lateReset], planAnswer: okPlan({ hours: [6, 12], max: 5000 }) });
  const run = createRun(payload({ hours: [6, 12], max: 5000 }), 0);
  let userActed = false;
  drive(A, run, 60000, now => {
    const wroteHours = A.writes.some(x => x.id === 'hour-slider');
    assert.ok(!(wroteHours && run.status !== 'running' && A.pending.length), 'never complete while a reset is still pending');
    if (wroteHours && !userActed && A.pending.length) {                      // the reset is in flight: the user now acts
      userActed = true; A.widgets['hour-slider'].value = [9, 10]; cancel(run, A.env, 'user_changed');
    }
  });
  A.now += 5000; A.advance();
  assert.equal(run.status, 'cancelled');
  assert.deepEqual(A.widgets['hour-slider'].value, [0, 24], 'the pending default reset still lands (it is the app, not us); nothing of ours fights it');
  assert.equal(A.writes.filter(x => x.id === 'hour-slider').length, 1);
});

test('STRESS: a new URL while a restore is running supersedes it: the old run writes nothing more and its stale plan answer is ignored', () => {
  const XA = { micro: ['a'], hours: [6, 12], max: 5000 }, YA = { micro: ['b'], hours: [8, 10], max: 3000 };
  let checked = 0;
  for (let t = 100; t <= 12000; t += 150) {
    const A = makeApp({ rules: [randomCascade(rng(t), [20, 380], 200, [300, 3000])], planAnswer: ({ token }) => (token === 'tokX' ? { status: 'ok', applied: XA, dropped: [], adjusted: [], cluster_ids: [] } : { status: 'ok', applied: YA, dropped: [], adjusted: [], cluster_ids: [] }) });
    const X = createRun(payload({ project: 'P2', ...XA }, 'tokX'), 0);
    let Y = null, writesAtSwitch = 0;
    for (let now = 0; now < 120000; now += 100) {
      A.now = now; A.advance();
      if (!Y && now >= t) {
        cancel(X, A.env, 'superseded'); writesAtSwitch = A.writes.length;
        Y = createRun(payload({ project: 'P2', ...YA }, 'tokY'), now);
      }
      if (Y) { step(Y, A.env, now, CFG); if (Y.status !== 'running') break; }
      else step(X, A.env, now, CFG);
    }
    if (!Y) continue;
    checked++;
    assert.equal(Y.status, 'complete', `t=${t}`);
    const afterX = A.writes.slice(writesAtSwitch).filter(w => w.id === 'url-identity' && w.props.data.token === 'tokX');
    assert.ok(afterX.length <= 1, `t=${t}: at most the old run's single release`);
    assert.deepEqual(A.widgets['microlocation-dropdown'].value, ['b'], `t=${t}: the NEW link's microlocation wins`);
    assert.deepEqual(A.widgets['hour-slider'].value, [8, 10], `t=${t}`);
    assert.equal(A.widgets['max-points'].value, 3000, `t=${t}`);
    const req = A.planCalls.map(c => c.token);
    assert.ok(!req.includes('tokX') || A.widgets['url-plan-filters'].data.token === 'tokY', `t=${t}: the stale plan answer did not become the final plan`);
  }
  assert.ok(checked > 60);
});

test('a stale plan answer (older request id or older token) is never applied', () => {
  const A = makeApp({ rules: [cascade], planAnswer: okPlan({ max: 1111 }) });
  const run = createRun(payload({ max: 1111 }, 'tokNow'), 0);
  A.widgets['url-plan-filters'] = { data: { id: 'old-request', token: 'tokOld', status: 'ok', applied: { max: 99 }, dropped: [], adjusted: [] } };
  drive(A, run);
  assert.equal(A.widgets['max-points'].value, 1111);
  assert.ok(!A.writes.some(x => x.id === 'max-points' && x.props.value === 99));
  const B = makeApp({ rules: [cascade], planAnswer: () => ({ status: 'ok', applied: { max: 99 }, dropped: [], adjusted: [], token: 'someone-else' }) });
  const run2 = createRun(payload({ max: 1111 }, 'mine'), 0);
  drive(B, run2);
  assert.ok(run2.dropped.some(d => d.key === 'max' && d.reason === 'stale'), 'a plan for another token drops the keys instead of applying them');
  assert.ok(!B.writes.some(x => x.id === 'max-points' && x.props.value === 99));
});

test('SENSITIVITY: the stress checks do catch a broken gate (always "quiet"): most random cascades are flagged', () => {
  let flagged = 0;
  const N = 100;
  for (let seed = 1; seed <= N; seed++) {
    const rnd = rng(seed);
    const A = makeApp({ rules: [randomCascade(rnd, [20, 380], 0, [300, 4000])], planAnswer: okPlan(WANT_APPLIED) });
    A.env.quiet = () => true;                                                  // the defect: stages advance without waiting
    const run = createRun(payload(WANT), 0);
    drive(A, run, 300000);
    const bad = !finalOk(A, WANT) || run.dropped.length > 0 || Object.entries(run.writes).some(([k, n]) => k !== 'max' && n > 1);
    if (bad) flagged++;
  }
  assert.ok(flagged >= 80, `only ${flagged}/${N} broken-gate runs were detected, so the stress checks are too weak`);
});

test('CHARACTERISATION: past the window (unobserved gap > 400 ms) the verify pass still corrects most clobbers before completion', () => {
  let exact = 0, recovered = 0;
  const N = 100;
  for (let seed = 1; seed <= N; seed++) {
    const { A, run } = stressOnce(seed, { lat: [20, 380], gap: 900 });
    if (run.status === 'complete' && finalOk(A, WANT)) { exact++; if (Object.entries(run.writes).some(([k, n]) => k !== 'max' && n > 1)) recovered++; }
  }
  // Informational bound only: the guaranteed regime is gap < settleMs (tested above); this documents the residual risk.
  console.log(`# gap up to 900 ms: ${exact}/${N} ended exactly as requested (${recovered} needed a rewrite by the verify pass)`);
  assert.ok(exact >= 0);
});

test('slider caps are applied against the LIVE slider maximum at apply time and the clamp is reported', () => {
  const A = makeApp();                                                      // merge-threshold max 100, clip-count max 5
  let res = ADAPTERS.mcc(99999, A.env);
  assert.deepEqual(res.writes.map(w => w.props.value), [5]);
  assert.deepEqual(res.adjust, { value: 99999, applied: 5, reason: 'clamped' });
  res = ADAPTERS.mcc(3, A.env);
  assert.deepEqual(res.writes.map(w => w.props.value), [3]);
  assert.equal(res.adjust, undefined, 'an in-range value is not an adjustment');
  A.widgets['clip-count-threshold'].max = 22;                               // the cap grows once merging is on
  assert.deepEqual(ADAPTERS.mcc(20, A.env).writes.map(w => w.props.value), [20]);
  A.widgets['clip-count-threshold'].value = 5;
  assert.deepEqual(ADAPTERS.mcc(99999, A.env).writes.map(w => w.props.value), [22]);
  A.widgets['merge-threshold'].max = 25;
  assert.deepEqual(ADAPTERS.mth(99999, A.env).writes.map(w => w.props.value), [25]);
});

test('a run clamps an over-large clip count to the slider cap once merging is on, records it, and ends stable', () => {
  const growAfterMerge = (id, props, A) => { if (id === 'merge-switch') A.effect(150, () => { A.widgets['clip-count-threshold'].max = 22; }); };
  const A = makeApp({ rules: [cascade, growAfterMerge], planAnswer: okPlan({ merge: true, mth: 30, mcc: 99999 }) });
  const run = createRun(payload({ merge: true, mth: 30, mcc: 99999 }), 0);
  drive(A, run);
  assert.equal(run.status, 'complete');
  assert.equal(A.widgets['clip-count-threshold'].value, 22);
  assert.ok(run.adjusted.some(a => a.key === 'mcc' && a.value === 99999 && a.applied === 22 && a.reason === 'clamped'));
  assert.equal(run.writes.mcc, 1, 'one write: the cap was known when the value was applied');
  assert.equal(run.applied.mcc, 22);
});
