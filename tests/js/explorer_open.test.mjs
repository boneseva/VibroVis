// Node tests for assets/explorer_open.js: resolving a VibroVis clip to an Explorer recording page with only the
// Explorer's existing API (a fake one here).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const api = createRequire(import.meta.url)(path.join(here, '..', '..', 'assets', 'explorer_open.js'));
const { parseLink, pickDeployment, resolve, targetUrl } = api;

const HREF = '/explorer/go?dataset=citimon&path=2026_03-C20/20260228_211000&t=12.35';
const META = {
  deployments: [
    { id: 7, folder: '2026_03-C20', recorder: 'C20', first: '2026-02-09', last: '2026-03-21' },
    { id: 8, folder: '2026_03-C19', recorder: 'C19', first: '2026-03-01', last: '2026-03-21' },
    { id: 9, folder: '2026_04-C19', recorder: 'C19', first: '2026-03-22', last: '2026-04-12' },
  ],
  sources: [{ id: 1, kind: 'manual' }, { id: 2, kind: 'model', default_threshold: 0.4 }],
};

// Fake Explorer. `passages` is what api/passages returns, `day` what api/recording/<id> lists for the day.
const fake = ({ passages = [], day = [], unauthorized = false } = {}) => {
  const calls = [];
  const out = r => (unauthorized ? Promise.reject({ unauthorized: true }) : Promise.resolve(r));
  return {
    calls,
    get: u => { calls.push(['GET', u]); return out(u.includes('/api/meta') ? META : { day }); },
    post: (u, b) => { calls.push(['POST', u, b]); return out({ passages }); },
  };
};
const DAY = [{ id: 501, local_minute: 20 * 60 }, { id: 502, local_minute: 21 * 60 + 10 }, { id: 503, local_minute: 21 * 60 + 20 }];
const L = parseLink(HREF);

test('parseLink reads the descriptor href', () => {
  assert.deepEqual(L, { base: '/explorer', dataset: 'citimon', folder: '2026_03-C20', stem: '20260228_211000',
    date: '2026-02-28', minute: 21 * 60 + 10, t: 12.35 });
  assert.equal(parseLink('https://x.si/explorer/go?dataset=citimon&path=2026_03-C20/20260228_211000&t=-4').t, 0);
  assert.equal(parseLink('https://x.si/explorer/go?dataset=citimon&path=2026_03-C20/20260228_211000&t=-4').base, 'https://x.si/explorer');
  for (const bad of ['', null, '/explorer/', '/explorer/go?path=nofolder', '/explorer/go?path=a/b']) assert.equal(parseLink(bad), null, String(bad));
});

test('deployment: exact folder, else same recorder covering the day, else none', () => {
  assert.equal(pickDeployment(META, L).id, 7);
  const renamed = { ...L, folder: '2026_03-C19-2', date: '2026-04-01' };
  assert.equal(pickDeployment(META, renamed).id, 9);
  assert.equal(pickDeployment(META, { ...L, folder: '2026_03-C99' }), null);
  assert.equal(pickDeployment(META, { ...L, folder: 'Bioblitz' }), null);
});

test('a labelled recording at that minute resolves from the passages directly', async () => {
  const a = fake({ passages: [{ recording_id: 11, local_minute: 20 * 60 }, { recording_id: 42, local_minute: 21 * 60 + 10 }] });
  const r = await resolve(L, a);
  assert.equal(r.id, 42);
  assert.equal(a.calls.filter(c => c[0] === 'GET').length, 1, 'no day lookup needed');
  const body = a.calls.find(c => c[0] === 'POST')[2];
  assert.deepEqual([body.deployments, body.dates, body.v], [[7], ['2026-02-28'], 2]);
  assert.deepEqual(body.sources.map(s => [s.id, s.role]), [[1, 'include'], [2, 'include']]);
  assert.equal(r.id, 42);
  assert.deepEqual(body.sources.map(s => s.threshold), [0, 0], 'the lookup sees every label');
  const url = targetUrl(L, r);
  assert.equal(url, '/explorer/#/recording/42?t=12.35', 'recording URL is clean without overriding filters');
  const f = r.filters;
  assert.deepEqual([f.deployments, f.dates, f.hours, f.v], [[7], ['2026-02-28'], { mode: 'all' }, 2]);
  assert.deepEqual(f.sources, body.sources, 'the resolved filter is preserved for fallback');
});

test('an unlabelled recording resolves through the day list of another recording', async () => {
  const a = fake({ passages: [{ recording_id: 501, local_minute: 20 * 60 }], day: DAY });
  const r = await resolve(L, a);
  assert.equal(r.id, 502);
  assert.deepEqual(r.filters.dates, ['2026-02-28']);
  assert.match(a.calls.at(-1)[1], /\/explorer\/api\/recording\/501\?/);
});

test('nothing labelled that day, or the minute missing: the filtered passage list opens instead', async () => {
  for (const a of [fake({ passages: [] }), fake({ passages: [{ recording_id: 501, local_minute: 20 * 60 }], day: [DAY[0]] })]) {
    const r = await resolve(L, a);
    assert.equal(r.id, undefined);
    assert.deepEqual([r.filters.deployments, r.filters.dates, r.filters.hours], [[7], ['2026-02-28'], { mode: 'range', from: 1270, to: 1271 }]);
    const url = targetUrl(L, r);
    assert.match(url, /^\/explorer\/#\/passages\?f=/);
    assert.deepEqual(JSON.parse(decodeURIComponent(url.split('?f=')[1])), r.filters);
  }
});

test('unknown deployment gives the Explorer home; not logged in rejects as unauthorized', async () => {
  const r = await resolve({ ...L, folder: '2026_03-C99' }, fake());
  assert.deepEqual(r, { missing: true });
  assert.equal(targetUrl(L, r), '/explorer/');
  await assert.rejects(resolve(L, fake({ unauthorized: true })), e => e.unauthorized === true);
});
