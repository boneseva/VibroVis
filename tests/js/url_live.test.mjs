// Node tests for the live-URL / share side of assets/url_reconcile.js: the snapshot collector and the settled/debounce
// decision. (Serialisation is Python's job and is tested there; this side must never know the URL syntax.)
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const api = createRequire(import.meta.url)(path.join(here, '..', '..', 'assets', 'url_reconcile.js'));
const { collectSnapshot, syncDecide, canon } = api;
const source = readFileSync(path.join(here, '..', '..', 'assets', 'url_reconcile.js'), 'utf8');

const fakeEnv = (values = {}, rows = {}) => ({
  value: (id, prop) => (values[`${id}.${prop === undefined ? 'data' : prop}`]),
  rows: type => (rows[type] || []),
});
const V = {
  'project-dropdown.value': 'Citimon', 'location-dropdown.value': ['C08', 'C01'], 'model-dropdown.value': 'perchv2',
  'num-cluster-dropdown.value': 20, 'main-view-tabs.data': 'table-tab', 'color-mode-radio.value': 'manual',
  'raven-annotator-filter.value': [], 'raven-compare-mode.value': 'all', 'microlocation-dropdown.value': ['a'],
  'recorder-type-dropdown.value': ['r'], 'channel-checklist.value': [1], 'date-dropdown.data': ['2026-03-01'],
  'hour-slider.value': [6, 20], 'merge-switch.on': true, 'merge-threshold.value': 20, 'clip-count-threshold.value': 2,
  'max-points.value': 5000, 'priority-sampling-toggle.value': [], 'table-sort-store.data': { col: 'cluster_id', asc: false },
};

test('the snapshot holds the settled widget values and the sorted dataset key, and nothing else', () => {
  const snap = collectSnapshot(fakeEnv(V, {
    'cluster-checkbox': [{ id: { type: 'cluster-checkbox', index: 0 }, value: ['on'] }, { id: { type: 'cluster-checkbox', index: 1 }, value: [] }],
    'label-checkbox': [{ id: { type: 'label-checkbox', index: 'owl' }, value: ['on'] }, { id: { type: 'label-checkbox', index: 'Unlabeled' }, value: [] }],
  }), '?utm=1');
  assert.deepEqual(snap.key, ['Citimon', 'perchv2', ['C01', 'C08']]);
  assert.deepEqual(snap.clusters, { ids: [0, 1], ticked: [0] });
  assert.deepEqual(snap.labels, { rows: ['owl', 'Unlabeled'], ticked: ['owl'] });
  assert.equal(snap.view, 'table-tab'); assert.equal(snap.max, 5000); assert.deepEqual(snap.sort, { col: 'cluster_id', asc: false });
  assert.equal(snap.search, '?utm=1');
  // nothing transient or free text: the key set is exactly the documented one
  assert.deepEqual(Object.keys(snap).sort(), ['annot', 'ch', 'clusters', 'cmp', 'color', 'dates', 'hours', 'k', 'key', 'labels', 'loc', 'max',
    'mcc', 'merge', 'micro', 'model', 'mth', 'prio', 'project', 'rec', 'search', 'sort', 'view'].sort());
});

test('the client never serialises URLs: no query-string building and no owned key names in the sync code', () => {
  const sync = source.slice(source.indexOf('live URL and share: snapshot'), source.indexOf('Dash binding (browser only)'));
  assert.ok(!/encodeURIComponent|URLSearchParams|\.join\(['"]&['"]\)/.test(source.slice(source.indexOf('function collectSnapshot'))), 'no client-side serialiser');
  assert.ok(!/hours=|clusters=|dates=|micro=/.test(sync));
});

const settled = { decided: true, restoring: false, quiet: true, loaded: true };
test('a settled snapshot is sent once, only after it stayed unchanged for the debounce time', () => {
  const ss = { lastSent: null, pendingCanon: null, pendingSince: 0 };
  const snap = collectSnapshot(fakeEnv(V), '');
  assert.equal(syncDecide(ss, snap, settled, 1000, 700), 'none');          // first sight: start the debounce
  assert.equal(syncDecide(ss, snap, settled, 1500, 700), 'none');          // not long enough
  assert.equal(syncDecide(ss, snap, settled, 1800, 700), 'request');
  assert.equal(syncDecide(ss, snap, settled, 9000, 700), 'none');          // already sent: no repeat
});

test('a change restarts the debounce; a burst of changes yields ONE request for the last state', () => {
  const ss = { lastSent: null, pendingCanon: null, pendingSince: 0 };
  let sent = 0;
  for (let t = 0, hours = 6; t < 3000; t += 100, hours = (hours % 18) + 1) {          // slider being dragged
    const snap = collectSnapshot(fakeEnv({ ...V, 'hour-slider.value': [hours, 20] }), '');
    if (syncDecide(ss, snap, settled, t, 700) === 'request') sent++;
  }
  assert.equal(sent, 0, 'never while it keeps changing');
  const last = collectSnapshot(fakeEnv({ ...V, 'hour-slider.value': [9, 20] }), '');
  assert.equal(syncDecide(ss, last, settled, 3100, 700), 'none');
  assert.equal(syncDecide(ss, last, settled, 3900, 700), 'request');
});

test('nothing is sent while initialisation is undecided, a restore runs, the app is busy or the dataset is not loaded', () => {
  const snap = collectSnapshot(fakeEnv(V), '');
  for (const bad of [{ decided: false }, { restoring: true }, { quiet: false }, { loaded: false }]) {
    const ss = { lastSent: null, pendingCanon: null, pendingSince: 0 };
    const ctx = { ...settled, ...bad };
    for (let t = 0; t < 5000; t += 100) assert.equal(syncDecide(ss, snap, ctx, t, 700), 'none', JSON.stringify(bad));
    assert.equal(ss.pendingCanon, null);
  }
});

test('the search string and the transient parts do not influence the decision (no loop after replaceState)', () => {
  const a = collectSnapshot(fakeEnv(V), '?x=1'), b = collectSnapshot(fakeEnv(V), '?x=2&project=Citimon');
  assert.equal(canon(a), canon(b));
});
