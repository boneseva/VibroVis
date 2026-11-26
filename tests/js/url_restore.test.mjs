// Node tests for assets/url_restore.js (the canonical reducer + Dash binding). Run: node --test tests/js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const api = createRequire(import.meta.url)(path.join(here, '..', '..', 'assets', 'url_restore.js'));
const vectors = JSON.parse(readFileSync(path.join(here, '..', 'fixtures', 'url_restore_vectors.json'), 'utf8'));
const { initial, reduce, publicView, hasOwnedParams, makeEpochFactory, arbiter } = api;
const CFG = { pendingMs: 10000, runningMs: 60000, graceMs: 1500, tickMs: 500 };

const run = (events, epoch = 'E1') => {
  let s = initial(epoch, 0); const notes = [];
  for (const e of events) { const r = reduce(s, e, CFG); s = r.state; notes.push(r.note); }
  return { s, notes };
};
const resp = (epoch, id, now) => ({ type: 'response', epoch, payload: { token: id }, now });

// ---------------------------------------------------------------- phases and transitions
test('fast init: pending -> ready', () => assert.equal(run([resp('E1', 'a', 50)]).s.phase, 'ready'));
test('the four phases are explicit and distinct', () => {
  const seen = new Set();
  seen.add(initial('E1', 0).phase);
  seen.add(run([{ type: 'started', epoch: 'E1', now: 1 }]).s.phase);
  seen.add(run([resp('E1', 'a', 1)]).s.phase);
  seen.add(run([{ type: 'tick', now: 10000 }]).s.phase);
  assert.deepEqual([...seen].sort(), ['failed', 'pending', 'ready', 'running']);
});
test('slow but RUNNING init is not failed at the pending deadline', () => {
  assert.equal(run([{ type: 'started', epoch: 'E1', now: 100 }, { type: 'tick', now: 12000 }, { type: 'tick', now: 30000 }]).s.phase, 'running');
});
test('never started -> failed(never_started) exactly at the pending deadline', () => {
  assert.equal(run([{ type: 'tick', now: 9999 }]).s.phase, 'pending');
  const { s } = run([{ type: 'tick', now: 10000 }]);
  assert.deepEqual([s.phase, s.reason], ['failed', 'never_started']);
});
test('running too long -> failed(timeout), measured from the START of running', () => {
  assert.equal(run([{ type: 'started', epoch: 'E1', now: 5000 }, { type: 'tick', now: 64999 }]).s.phase, 'running');
  const { s } = run([{ type: 'started', epoch: 'E1', now: 5000 }, { type: 'tick', now: 65000 }]);
  assert.deepEqual([s.phase, s.reason], ['failed', 'timeout']);
});
test('callback ended without a response -> failed(no_response) only after the grace period', () => {
  assert.equal(run([{ type: 'started', epoch: 'E1', now: 100 }, { type: 'ended', epoch: 'E1', now: 200 }, { type: 'tick', now: 1000 }]).s.phase, 'running');
  const { s } = run([{ type: 'started', epoch: 'E1', now: 100 }, { type: 'ended', epoch: 'E1', now: 200 }, { type: 'tick', now: 1700 }]);
  assert.deepEqual([s.phase, s.reason], ['failed', 'no_response']);
});
test('a response racing the end-of-callback flag still wins inside the grace period', () => {
  const { s } = run([{ type: 'started', epoch: 'E1', now: 100 }, { type: 'ended', epoch: 'E1', now: 200 }, resp('E1', 't', 300), { type: 'tick', now: 5000 }]);
  assert.deepEqual([s.phase, s.payload], ['ready', { token: 't' }]);
});
test('an internal arbiter error fails the initialisation (consumers never hold forever)', () => {
  const { s } = run([{ type: 'fail', reason: 'arbiter_error' }]);
  assert.deepEqual([s.phase, s.reason, s.payload], ['failed', 'arbiter_error', null]);
});

// ---------------------------------------------------------------- terminal states are absorbing
test('A LATE RESPONSE CAN NOT REVIVE AN EXPIRED INITIALISATION', () => {
  const { s, notes } = run([{ type: 'tick', now: 10000 }, resp('E1', 'late', 10500)]);
  assert.deepEqual([s.phase, s.reason, s.payload], ['failed', 'never_started', null]);
  assert.equal(notes[1], 'ignored_response_after_failed');
});
test('a late response after a RUNNING timeout is ignored too', () => {
  const { s } = run([{ type: 'started', epoch: 'E1', now: 0 }, { type: 'tick', now: 60000 }, resp('E1', 'late', 61000)]);
  assert.deepEqual([s.phase, s.reason, s.payload], ['failed', 'timeout', null]);
});
test('a stale or duplicate response can not overwrite a ready state', () => {
  const { s, notes } = run([resp('E1', 'first', 10), resp('E1', 'second', 20), resp('E0', 'stale', 30)]);
  assert.deepEqual(s.payload, { token: 'first' });
  assert.deepEqual(notes.slice(1), ['ignored_response_after_ready', 'ignored_response_after_ready']);
});
test('a response for another epoch is never accepted while pending', () => {
  const { s, notes } = run([resp('E9', 'x', 10)]);
  assert.deepEqual([s.phase, notes[0]], ['pending', 'ignored_stale_epoch']);
});
test('ticks after a terminal phase change nothing', () => {
  assert.equal(run([resp('E1', 'a', 5), { type: 'tick', now: 1e9 }]).s.phase, 'ready');
});

// ---------------------------------------------------------------- a NEW initialisation invalidates older responses and timers
test('restart begins a fresh pending state with a new epoch from ANY phase', () => {
  for (const before of [[], [{ type: 'started', epoch: 'E1', now: 1 }], [resp('E1', 'a', 1)], [{ type: 'tick', now: 10000 }]]) {
    const { s } = run([...before, { type: 'restart', epoch: 'E2', now: 20000, hadOwnedParams: true }]);
    assert.deepEqual([s.phase, s.epoch, s.payload, s.reason, s.hadOwnedParams], ['pending', 'E2', null, null, true]);
    assert.ok(s.attempt >= 2);
  }
});
test('after a restart an OLD epoch response is ignored, the NEW one is accepted', () => {
  const { s, notes } = run([{ type: 'started', epoch: 'E1', now: 1 }, { type: 'restart', epoch: 'E2', now: 100 }, resp('E1', 'old', 200), resp('E2', 'new', 300)]);
  assert.deepEqual(notes.slice(2), ['ignored_stale_epoch', 'ready']);
  assert.deepEqual(s.payload, { token: 'new' });
});
test('an old response that arrives after the new one is ready can not overwrite it', () => {
  const { s } = run([{ type: 'restart', epoch: 'E2', now: 100 }, resp('E2', 'new', 150), resp('E1', 'old', 400)]);
  assert.deepEqual(s.payload, { token: 'new' });
});
test('a late response for an expired epoch can not revive it even after a restart elsewhere', () => {
  const { s } = run([{ type: 'tick', now: 10000 }, { type: 'restart', epoch: 'E2', now: 11000 }, resp('E1', 'late-from-the-failed-attempt', 11500)]);
  assert.deepEqual([s.phase, s.epoch, s.payload], ['pending', 'E2', null]);
});
test('deadlines restart with the new epoch: old timers do not count', () => {
  const { s } = run([{ type: 'restart', epoch: 'E2', now: 9000 }, { type: 'tick', now: 18999 }]);
  assert.equal(s.phase, 'pending');
  assert.equal(run([{ type: 'restart', epoch: 'E2', now: 9000 }, { type: 'tick', now: 19000 }]).s.reason, 'never_started');
});
test('stale started / ended flags of the previous epoch are ignored', () => {
  const { notes } = run([{ type: 'restart', epoch: 'E2', now: 10 }, { type: 'started', epoch: 'E1', now: 20 }, { type: 'ended', epoch: 'E1', now: 30 }]);
  assert.deepEqual(notes.slice(1), ['ignored_stale_started', 'ignored_stale_ended']);
});
test('restarting with the SAME epoch is refused', () => {
  assert.equal(run([{ type: 'restart', epoch: 'E1', now: 5 }]).notes[0], 'ignored_restart_same_epoch');
});
test('epochs from the factory are unique, also across two runtimes', () => {
  const a = makeEpochFactory(), b = makeEpochFactory();
  const all = [a(), a(), a(), b(), b()];
  assert.equal(new Set(all).size, all.length);
});

// ---------------------------------------------------------------- owned-parameter detection (shared vectors with Python)
test('hasOwnedParams agrees with every shared vector', () => {
  for (const c of vectors.owned_params) assert.equal(hasOwnedParams(c.search, vectors.owned_keys), c.expect, c.name);
});
test('the vector file lists the owned keys and enough cases', () => {
  assert.ok(vectors.owned_keys.includes('project') && vectors.owned_keys.includes('v'));
  assert.ok(vectors.owned_params.length >= 12);
});

// ---------------------------------------------------------------- the Dash binding, with a fake dash_clientside
const NU = Symbol('no_update');
let clock = 0;
function withFakeDash(fn) {
  delete globalThis.__vvUrlInit;
  const savedPerf = Object.getOwnPropertyDescriptor(globalThis, 'performance');
  Object.defineProperty(globalThis, 'performance', { value: { now: () => clock }, configurable: true, writable: true });
  globalThis.dash_clientside = { no_update: NU, callback_context: { triggered: [] }, set_props() {} };
  try { return fn(); } finally { delete globalThis.dash_clientside; delete globalThis.__vvUrlInit; delete globalThis.__vvArbiterAlive; if (savedPerf) Object.defineProperty(globalThis, 'performance', savedPerf); }
}
const call = (trig, running, n, search, config, response) => {
  globalThis.dash_clientside.callback_context = { triggered: trig.map(prop_id => ({ prop_id })) };
  return arbiter(running, n, search, config, response);
};
const BCFG = { ...CFG, ownedKeys: vectors.owned_keys };

test('binding: first run begins epoch 1, publishes pending, requests the server, enables the timer', () => withFakeDash(() => {
  clock = 0;
  const [pub, req, tick] = call([], false, 0, '?project=Citimon', BCFG, null);
  assert.deepEqual([pub.phase, pub.hadOwnedParams, pub.attempt], ['pending', true, 1]);
  assert.deepEqual([typeof req.epoch, req.search], ['string', '?project=Citimon']);
  assert.equal(tick, false);
  assert.equal(globalThis.__vvArbiterAlive, true);
}));
test('binding: no owned parameters -> hadOwnedParams false (no notice later)', () => withFakeDash(() => {
  const [pub] = call([], false, 0, '?utm=1', BCFG, null);
  assert.equal(pub.hadOwnedParams, false);
  assert.equal(call([], false, 1, '', BCFG, null)[0], NU);
}));
test('binding: running flag -> running; the answer is read as STATE on the next tick -> ready; timer stops', () => withFakeDash(() => {
  clock = 0; const [, req] = call([], false, 0, '', BCFG, null);
  clock = 100; assert.equal(call(['url-init-running.data'], true, 0, '', BCFG, null)[0].phase, 'running');
  clock = 600; const [pub, , tick] = call(['url-init-tick.n_intervals'], true, 1, '', BCFG, { epoch: req.epoch, payload: { status: 'none' } });
  assert.deepEqual([pub.phase, pub.payload, tick], ['ready', { status: 'none' }, true]);
}));
test('binding: a late answer after the timeout is ignored and publishes nothing', () => withFakeDash(() => {
  clock = 0; const [, req] = call([], false, 0, '?k=1', BCFG, null);
  clock = 100; call(['url-init-running.data'], true, 0, '?k=1', BCFG, null);
  clock = 60200; const [failedPub] = call(['url-init-tick.n_intervals'], true, 1, '?k=1', BCFG, null);
  assert.deepEqual([failedPub.phase, failedPub.reason, failedPub.hadOwnedParams], ['failed', 'timeout', true]);   // info kept for a later notice
  clock = 61000; const [lateOut] = call(['url-init-tick.n_intervals'], false, 2, '?k=1', BCFG, { epoch: req.epoch, payload: { status: 'active', token: 'late' } });
  assert.equal(lateOut, NU);
}));
test('binding: a changed URL starts a NEW attempt with a new epoch; the old attempt\'s answer is ignored', () => withFakeDash(() => {
  clock = 0; const [, req1] = call([], false, 0, '?a=1', BCFG, null);
  clock = 50; const [pub2, req2] = call(['url.search'], false, 0, '?project=X', BCFG, null);
  assert.notEqual(req1.epoch, req2.epoch);
  assert.deepEqual([pub2.phase, pub2.attempt, pub2.hadOwnedParams], ['pending', 2, true]);
  clock = 100; const [out] = call(['url-init-tick.n_intervals'], false, 1, '?project=X', BCFG, { epoch: req1.epoch, payload: { token: 'old' } });
  assert.equal(out, NU);
  clock = 150; const [ready] = call(['url-init-tick.n_intervals'], false, 2, '?project=X', BCFG, { epoch: req2.epoch, payload: { token: 'new' } });
  assert.deepEqual([ready.phase, ready.payload], ['ready', { token: 'new' }]);
}));
test('binding: an unchanged search does not restart initialisation', () => withFakeDash(() => {
  call([], false, 0, '?a=1', BCFG, null);
  const out = call(['url.search'], false, 0, '?a=1', BCFG, null);
  assert.equal(out[1], NU);
}));
test('binding: an error inside the arbiter publishes failed(arbiter_error)', () => withFakeDash(() => {
  clock = 0; call([], false, 0, '', BCFG, null);
  globalThis.dash_clientside.callback_context = { triggered: [{ prop_id: 'url-init-tick.n_intervals' }] };
  const [pub, , tick] = arbiter(false, 1, '', BCFG, { get epoch() { throw new Error('boom'); } });
  assert.deepEqual([pub.phase, pub.reason, tick], ['failed', 'arbiter_error', true]);
}));

// ---------------------------------------------------------------- bounded EXHAUSTIVE model check, restarts included
test('model check: invariants hold for every sequence up to length 5', () => {
  const alphabet = [
    { type: 'started', epoch: 'E1' }, { type: 'ended', epoch: 'E1' }, { type: 'started', epoch: 'E2' }, { type: 'ended', epoch: 'E2' },
    { type: 'response', epoch: 'E1', payload: { id: 'A1' } }, { type: 'response', epoch: 'E2', payload: { id: 'A2' } },
    { type: 'response', epoch: 'EX', payload: { id: 'STALE' } },
    { type: 'restart', epoch: 'E2' }, { type: 'restart', epoch: 'E3' },
    { type: 'tick', dt: 500 }, { type: 'tick', dt: 11000 }, { type: 'tick', dt: 61000 },
  ];
  const isTerminal = s => s.phase === 'ready' || s.phase === 'failed';
  let transitions = 0, ready = 0, failed = 0, restarts = 0;
  const rec = (state, depth, now) => {
    if (depth === 5) return;
    for (const a of alphabet) {
      const t = now + (a.dt || 10), ev = Object.assign({}, a, { now: t });
      const before = state, after = reduce(before, ev, CFG).state;
      transitions++;
      const isRestart = ev.type === 'restart' && ev.epoch !== before.epoch;
      if (isRestart) {                                          // I5: only a restart changes the epoch, to a FRESH pending state
        restarts++;
        assert.deepEqual([after.phase, after.epoch, after.payload, after.reason, after.since], ['pending', ev.epoch, null, null, t]);
        assert.ok(after.attempt > before.attempt);
      } else {
        assert.equal(after.epoch, before.epoch);
        if (isTerminal(before)) assert.deepEqual(after, before);                             // I1 terminal phases are absorbing
        if (['started', 'ended', 'response'].includes(ev.type) && ev.epoch !== before.epoch) assert.deepEqual(after, before);   // I4 stale events are inert
        if (after.phase === 'ready' && before.phase !== 'ready')                             // I2 ready only via a current-epoch response
          assert.ok(ev.type === 'response' && ev.epoch === before.epoch && ['pending', 'running'].includes(before.phase));
        if (before.phase === 'ready') assert.deepEqual(after.payload, before.payload);        // I3 the accepted payload never changes
      }
      if (after.phase === 'failed') assert.equal(after.payload, null);
      if (after.phase === 'ready') assert.equal(after.reason, null);
      if (after.phase === 'ready') ready++;
      if (after.phase === 'failed') failed++;
      rec(after, depth + 1, t);
    }
  };
  rec(initial('E1', 0), 0, 0);
  assert.ok(transitions > 250000, `explored ${transitions}`);
  assert.ok(ready > 0 && failed > 0 && restarts > 0);
  console.log(`      explored ${transitions.toLocaleString()} transitions (${ready.toLocaleString()} ready, ${failed.toLocaleString()} failed, ${restarts.toLocaleString()} restarts): all invariants hold`);
});
