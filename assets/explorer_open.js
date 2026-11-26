/* "Open in Explorer": finds a clip's recording in the Citimon Explorer from the browser, using only the Explorer's
 * existing API and the user's own Explorer login (same origin, same cookie). Nothing on the Explorer's side is needed.
 *
 * The link's href (built in Python, explorer_link.py) is a descriptor:
 *     <base>/go?dataset=citimon&path=<folder>/<YYYYMMDD_HHMMSS>&t=<seconds>
 * A click is intercepted here and resolved to  <base>/#/recording/<id>?t=<seconds>  (the Explorer's own page):
 *   1. GET  api/meta                  -> the deployment (folder, else recorder + date) and the sources;
 *   2. POST api/passages              -> labelled passages of that deployment on that day (every source), the
 *                                        recording's id when it has labels at that minute;
 *   3. GET  api/recording/<any id>    -> its "day" list holds EVERY recording of the day with its id, so unlabelled
 *                                        recordings resolve too (when anything is labelled that day);
 *   4. otherwise the Explorer's passage list filtered to that deployment, day and minute.
 * The recording page also gets the filter "this deployment, this day" (`f`), so going back to the passage list shows
 * the same recorder and day instead of the last search.
 * Not logged in to the Explorer (401): its login page opens; log in there and click the link again.
 * Browser: the script is a classic file (Dash loads assets/*.js); Node tests use module.exports.
 */
(function (root) {
  'use strict';

  var STEM = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/;
  var FOLDER = /^(\d{4})[-_](\d{2})-([A-Za-z]\d+)(?:-(\d))?$/;

  function parseLink(href) {
    var m = /^(.*)\/go\?(.*)$/.exec(String(href || ''));
    if (!m) return null;
    var q = {};
    m[2].split('&').forEach(function (kv) {
      var i = kv.indexOf('=');
      if (i > 0) q[decodeURIComponent(kv.slice(0, i))] = decodeURIComponent(kv.slice(i + 1));
    });
    var parts = String(q.path || '').split('/');
    var sm = parts.length === 2 && STEM.exec(parts[1]);
    if (!sm) return null;
    var t = parseFloat(q.t);
    return {
      base: m[1], dataset: q.dataset || 'citimon', folder: parts[0], stem: parts[1],
      date: sm[1] + '-' + sm[2] + '-' + sm[3], minute: (+sm[4]) * 60 + (+sm[5]),
      t: isFinite(t) && t >= 0 ? t : 0,
    };
  }

  // Deployment of the clip: the folder itself, else (renamed folders) the same recorder whose dates cover the day.
  function pickDeployment(meta, link) {
    var deps = (meta && meta.deployments) || [];
    var exact = deps.filter(function (d) { return d.folder === link.folder; });
    if (exact.length) return exact[0];
    var fm = FOLDER.exec(link.folder);
    if (!fm) return null;
    var rec = fm[3].toUpperCase();
    var cover = deps.filter(function (d) {
      return String(d.recorder).toUpperCase() === rec && String(d.first).slice(0, 10) <= link.date &&
        link.date <= String(d.last).slice(0, 10);
    });
    return cover.length ? cover[0] : null;
  }

  // Threshold 0 on purpose: the lookup must see every label, and the list shown to the person uses the SAME filter, so
  // it can never be empty when the lookup found something (the Explorer's own default thresholds can leave a day empty).
  function sourcesOf(meta, role) {
    return ((meta && meta.sources) || []).map(function (s) { return { id: s.id, role: role, threshold: 0 }; });
  }

  // The Explorer's own filter object (version 2), restricted to one deployment and day.
  function filters(meta, dep, link, extra) {
    var f = {
      dataset: link.dataset, sites: [], deployments: [dep.id], dates: [link.date], hours: { mode: 'all' },
      taxa: [], sources: sourcesOf(meta, 'include'), include_unsure: true, sort: 'time', gap: 3, v: 2,
    };
    for (var k in extra || {}) f[k] = extra[k];
    return f;
  }

  // What the Explorer shows around the recording / as the fallback list: this deployment and day (same sources as the
  // lookup, see sourcesOf), optionally one minute.
  function viewFilters(meta, dep, link, hours) {
    return filters(meta, dep, link, { hours: hours || { mode: 'all' } });
  }

  function minuteOf(x) { return x && x.local_minute; }

  // api = { get(url) -> Promise<json>, post(url, body) -> Promise<json> }; rejects with {unauthorized: true} on 401.
  // Resolves to { id } (the recording) or { filtersUrl } (nothing to open directly) or { missing: true }.
  function resolve(link, api) {
    var base = link.base;
    return api.get(base + '/api/meta?dataset=' + encodeURIComponent(link.dataset)).then(function (meta) {
      var dep = pickDeployment(meta, link);
      if (!dep) return { missing: true };
      var body = filters(meta, dep, link, { offset: 0, limit: 500 });
      return api.post(base + '/api/passages', body).then(function (res) {
        var list = (res && res.passages) || [];
        var hit = list.filter(function (p) { return minuteOf(p) === link.minute; })[0];
        var minuteOnly = { mode: 'range', from: link.minute, to: link.minute + 1 };
        if (hit) return { id: hit.recording_id, filters: viewFilters(meta, dep, link) };
        if (!list.length) return { filters: viewFilters(meta, dep, link, minuteOnly) };
        return api.get(base + '/api/recording/' + list[0].recording_id + '?sources=&thresholds=&merge=0').then(function (r) {
          var day = ((r && r.day) || []).filter(function (x) { return minuteOf(x) === link.minute; })[0];
          if (day) return { id: day.id, filters: viewFilters(meta, dep, link) };
          return { filters: viewFilters(meta, dep, link, minuteOnly) };
        });
      });
    });
  }

  function targetUrl(link, result) {
    var f = result && result.filters ? encodeURIComponent(JSON.stringify(result.filters)) : '';
    if (result && result.id != null) return link.base + '/#/recording/' + result.id + '?t=' + link.t.toFixed(2) + (f ? '&f=' + f : '');
    if (f) return link.base + '/#/passages?f=' + f;
    return link.base + '/';
  }

  var api = { parseLink: parseLink, pickDeployment: pickDeployment, filters: filters, resolve: resolve, targetUrl: targetUrl };

  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }

  function http(method, url, body) {
    return fetch(url, {
      method: method, credentials: 'same-origin',
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then(function (r) {
      if (r.status === 401) return Promise.reject({ unauthorized: true });
      if (!r.ok) return Promise.reject(new Error('Explorer answered ' + r.status));
      return r.json();
    });
  }
  var browserApi = { get: function (u) { return http('GET', u); }, post: function (u, b) { return http('POST', u, b); } };

  function onClick(e) {
    var a = e.target && e.target.closest ? e.target.closest('#explorer-link') : null;
    if (!a || (e.type === 'click' && (e.ctrlKey || e.metaKey || e.shiftKey)) || (e.type === 'auxclick' && e.button !== 1)) return;
    var link = parseLink(a.getAttribute('href'));
    if (!link) return;
    e.preventDefault();
    // Open the tab now (a popup opened after an async call is blocked); it is pointed at the page once known.
    var w = window.open('', '_blank');
    try { if (w) w.document.body.textContent = 'Opening in the Citimon Explorer…'; } catch (err) { /* ignore */ }
    var go = function (url) { if (w) { w.location.href = url; } else { window.location.assign(url); } };
    resolve(link, browserApi).then(function (res) { go(targetUrl(link, res)); }).catch(function (err) {
      if (!(err && err.unauthorized)) console.error('Open in Explorer failed:', err);
      go(link.base + '/');
    });
  }
  document.addEventListener('click', onClick);
  document.addEventListener('auxclick', onClick);
})(typeof window !== 'undefined' ? window : globalThis);
