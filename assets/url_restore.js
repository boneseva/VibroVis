/**
 * url_restore.js: URL initialisation state machine (Phase 2.1 of URL sharing). THE canonical source of the reducer.
 *
 * This file only decides "has the URL been examined yet, and what did the server answer?". It restores nothing itself:
 * the reconciler (url_reconcile.js) applies the `active` payload it publishes.
 *
 * Public phases (explicit, and the only ones consumers may see):
 *
 *   pending  initialisation has not started (the server callback has not begun)
 *   running  the server callback `url_init` is executing
 *   ready    the server answered for the CURRENT epoch                       (terminal)
 *   failed   never started / timed out / ended without an answer / arbiter error (terminal)
 *
 * Rules (all covered by tests/js/url_restore.test.mjs, including an exhaustive bounded model check):
 *   - ready and failed are ABSORBING: a late response, tick or flag change can never revive or alter them;
 *   - a response is accepted only in pending/running and only for the current epoch;
 *   - `restart` (a NEW initialisation, e.g. the URL changed) begins a fresh epoch from ANY phase and thereby invalidates
 *     every older response and every older deadline: events carry the epoch they belong to and stale ones are ignored;
 *   - the server's answer is only an EVENT for this reducer; it never writes the public state.
 *
 * The Dash binding (`dash_clientside.url_restore.arbiter`) is the ONLY writer of the public `url-init` Store. It reads the
 * server answer as State, never as an Input: an Input that is the output of a running callback would make Dash hold the
 * whole arbiter until that callback finished, blinding it to the deadlines.
 */
(function (root, factory) {
  const api = factory(root);
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (root) root.VibrovisUrlRestore = api;
})(typeof self !== 'undefined' ? self : (typeof globalThis !== 'undefined' ? globalThis : this), function (root) {
  'use strict';

  const DEFAULT_CFG = { pendingMs: 10000, runningMs: 60000, graceMs: 1500, tickMs: 500, ownedKeys: [] };

  function initial(epoch, now, extras) {
    extras = extras || {};
    return {
      phase: 'pending', epoch, attempt: extras.attempt || 1, since: now, running: false, endedAt: null,
      reason: null, payload: null, hadOwnedParams: extras.hadOwnedParams === undefined ? false : extras.hadOwnedParams,
    };
  }

  /** Pure transition function: (state, event, cfg) -> {state, changed, note}. */
  function reduce(s, e, cfg) {
    cfg = Object.assign({}, DEFAULT_CFG, cfg || {});
    const keep = note => ({ state: s, changed: false, note });
    if (e.type === 'restart') {                       // handled before the terminal guard: a new initialisation
      if (e.epoch === s.epoch) return keep('ignored_restart_same_epoch');
      return {
        state: initial(e.epoch, e.now, { attempt: e.attempt || (s.attempt || 1) + 1, hadOwnedParams: e.hadOwnedParams }),
        changed: true, note: 'restart',
      };
    }
    if (s.phase === 'ready' || s.phase === 'failed') return keep(`ignored_${e.type}_after_${s.phase}`);
    switch (e.type) {
      case 'started':
        if (e.epoch !== s.epoch) return keep('ignored_stale_started');
        if (s.phase === 'pending') return { state: Object.assign({}, s, { phase: 'running', since: e.now, running: true, endedAt: null }), changed: true, note: 'started' };
        return { state: Object.assign({}, s, { running: true, endedAt: null }), changed: false, note: 'running_again' };
      case 'ended':
        if (e.epoch !== s.epoch) return keep('ignored_stale_ended');
        return { state: Object.assign({}, s, { running: false, endedAt: e.now }), changed: false, note: 'ended' };
      case 'response':
        if (e.epoch !== s.epoch) return keep('ignored_stale_epoch');
        return { state: Object.assign({}, s, { phase: 'ready', payload: e.payload, running: false }), changed: true, note: 'ready' };
      case 'fail':                                       // an internal error of the arbiter itself
        return { state: Object.assign({}, s, { phase: 'failed', reason: e.reason || 'error', payload: null }), changed: true, note: 'failed_' + (e.reason || 'error') };
      case 'tick':
        if (s.phase === 'pending' && e.now - s.since >= cfg.pendingMs)
          return { state: Object.assign({}, s, { phase: 'failed', reason: 'never_started' }), changed: true, note: 'failed_never_started' };
        if (s.phase === 'running') {
          if (e.now - s.since >= cfg.runningMs)
            return { state: Object.assign({}, s, { phase: 'failed', reason: 'timeout' }), changed: true, note: 'failed_timeout' };
          if (!s.running && s.endedAt != null && e.now - s.endedAt >= cfg.graceMs)
            return { state: Object.assign({}, s, { phase: 'failed', reason: 'no_response' }), changed: true, note: 'failed_no_response' };
        }
        return keep('tick');
      default:
        return keep('unknown_event');
    }
  }

  const publicView = s => ({
    phase: s.phase, epoch: s.epoch, attempt: s.attempt, reason: s.reason, payload: s.payload, hadOwnedParams: s.hadOwnedParams,
  });

  /** True when the query string contains ANY VibroVis-owned key (mirrors url_state.has_owned_params; shared vectors). */
  function hasOwnedParams(search, ownedKeys) {
    let s = String(search == null ? '' : search);
    const hash = s.indexOf('#');
    if (hash >= 0) s = s.slice(0, hash);
    if (s.charAt(0) === '?') s = s.slice(1);
    const owned = new Set(ownedKeys || []);
    for (const seg of s.split('&')) {
      if (seg === '') continue;
      const eq = seg.indexOf('=');
      const rawKey = (eq < 0 ? seg : seg.slice(0, eq)).replace(/\+/g, ' ');
      let key;
      try { key = decodeURIComponent(rawKey); } catch (err) { key = rawKey; }
      if (owned.has(key)) return true;
    }
    return false;
  }

  /** Unique epochs: a per-runtime random prefix plus a counter, so two initialisations never share an epoch. */
  function makeEpochFactory() {
    const prefix = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
    let counter = 0;
    return () => `${prefix}-${++counter}`;
  }

  // ---- Dash binding (browser only) ------------------------------------------------------------------------------------
  function runtime() {
    return root.__vvUrlInit || (root.__vvUrlInit = {
      state: null, runningPrev: false, lastResp: null, lastSearch: null, forceRestart: false, attempt: 0, log: [],
      nextEpoch: makeEpochFactory(),
    });
  }

  /**
   * Inputs: running flag, timer, Location.search.  States: config, server answer.
   * Outputs: [public url-init, url-init-request, url-init-tick.disabled]. The only writer of all three.
   */
  function arbiter(running, nTick, search, config, response) {
    const dc = root.dash_clientside, NU = dc.no_update;
    const R = runtime();
    root.__vvArbiterAlive = true;
    const cfg = Object.assign({}, DEFAULT_CFG, config || {});
    const now = root.performance ? root.performance.now() : Date.now();
    try {
      const ctx = dc.callback_context;
      const trig = ctx && ctx.triggered ? ctx.triggered.map(t => t.prop_id) : [];
      const searchNow = search == null ? '' : String(search);
      let changed = false, request = NU, restarted = false;
      const apply = ev => {
        const r = reduce(R.state, ev, cfg);
        if (r.note !== 'tick') R.log.push({ t: Math.round(now), ev: ev.type, note: r.note, phase: r.state.phase, epoch: r.state.epoch });
        R.state = r.state;
        if (r.changed) changed = true;
      };

      // (re)initialisation: the first run, an explicit restart, or a changed URL
      if (R.state === null || R.forceRestart || (trig.includes('url.search') && searchNow !== R.lastSearch)) {
        const epoch = R.nextEpoch();
        const had = hasOwnedParams(searchNow, cfg.ownedKeys);
        R.attempt += 1;
        R.lastSearch = searchNow; R.forceRestart = false; R.runningPrev = false;
        if (R.state === null) {
          R.state = initial(epoch, now, { attempt: R.attempt, hadOwnedParams: had });
          R.log.push({ t: Math.round(now), ev: 'begin', note: 'begin', phase: 'pending', epoch });
          changed = true;
        } else {
          apply({ type: 'restart', epoch, now, attempt: R.attempt, hadOwnedParams: had });
        }
        request = { epoch, search: searchNow };
        restarted = true;
      }

      // the server answer is read as STATE; it is an event, never the state itself
      if (response && JSON.stringify(response) !== R.lastResp) {
        R.lastResp = JSON.stringify(response);
        apply({ type: 'response', epoch: response.epoch, payload: response.payload, now });
      }
      if (trig.includes('url-init-running.data')) {
        if (running === true) apply({ type: 'started', epoch: R.state.epoch, now });
        else if (running === false && R.runningPrev) apply({ type: 'ended', epoch: R.state.epoch, now });
        R.runningPrev = (running === true);
      }
      apply({ type: 'tick', now });                          // every invocation also checks the deadlines

      const terminal = R.state.phase === 'ready' || R.state.phase === 'failed';
      const tickDisabled = terminal ? true : (restarted ? false : NU);
      return [changed ? publicView(R.state) : NU, request, tickDisabled];
    } catch (err) {                                          // never leave consumers holding forever
      const s = R.state || initial('error', now, {});
      R.state = Object.assign({}, s, { phase: 'failed', reason: 'arbiter_error', payload: null });
      R.log.push({ t: Math.round(now), ev: 'error', note: String(err && err.message || err), phase: 'failed', epoch: s.epoch });
      return [publicView(R.state), NU, true];
    }
  }

  /** Begin a NEW initialisation (invalidates older answers and timers). Browser only; used when the URL changes. */
  function restart() {
    const dc = root.dash_clientside, R = runtime();
    R.forceRestart = true;
    let n = 0;
    try { n = root.dash_component_api.getLayout('url-init-tick').props.n_intervals || 0; } catch (err) { /* no-op */ }
    dc.set_props('url-init-tick', { n_intervals: n + 1, disabled: false });
  }

  function debug() {
    const R = runtime();
    return { state: R.state ? publicView(R.state) : null, log: R.log.slice(), attempt: R.attempt };
  }

  if (root && typeof root.document !== 'undefined') {
    root.dash_clientside = root.dash_clientside || {};
    root.dash_clientside.url_restore = { arbiter };
  }
  return { DEFAULT_CFG, initial, reduce, publicView, hasOwnedParams, makeEpochFactory, arbiter, restart, debug };
});
