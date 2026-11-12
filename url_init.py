"""
Static pieces of the URL-initialisation skeleton (Phase 2.1). Standard library + limits + url_state only.

The initialisation PARSES the URL and records what it found; it does not restore any value itself (the reconciler in
``assets/url_reconcile.js`` does, from the ``active`` payload). The state machine lives in ``assets/url_restore.js``
(the single canonical source).
"""
import json
import sys

import limits
import url_state

# Public state of the initialisation, as it is in the layout BEFORE the browser-side arbiter has run. It is deliberately
# "pending": consumers that would load data hold on pending/running (see url_state.init_should_hold).
#   hadOwnedParams  True/False once known; None means "unknown" (only if the arbiter itself could not run). A later
#                   notice shows a warning for a FAILED initialisation unless hadOwnedParams is False.
INITIAL_PUBLIC_STATE = {"phase": "pending", "epoch": None, "attempt": 0, "reason": None, "payload": None,
                        "hadOwnedParams": False}

ARBITER_MISSING_MARGIN_MS = 5000


_CACHED = {}


def client_config(environ=None):
    """
    Deadlines (validated, safe defaults) plus the VibroVis-owned keys, for the browser-side arbiter. With the default
    environment the result is computed (and any invalid setting reported) once per process.
    """
    if environ is None and "cfg" in _CACHED:
        return dict(_CACHED["cfg"])
    cfg = limits.url_init_config(environ)
    restore_ms, restore_invalid = limits.url_restore_timeout_ms(environ)
    invalid = list(cfg["invalid"]) + restore_invalid
    if invalid:
        print(f"[url_init] ignoring invalid configuration {invalid}; using safe defaults where needed",
              file=sys.stderr)
    out = {"pendingMs": cfg["pendingMs"], "runningMs": cfg["runningMs"], "graceMs": cfg["graceMs"],
           "tickMs": cfg["tickMs"], "restoreMs": restore_ms, "ownedKeys": sorted(url_state.OWNED_KEYS)}
    if environ is None:
        _CACHED["cfg"] = dict(out)
    return out


def fallback_script(config=None):
    """
    Inline <script> that keeps the app usable if the arbiter never runs at all (asset blocked or broken): after the
    pending deadline plus a margin, and only if the arbiter never marked itself alive, it publishes
    ``failed(arbiter_missing)`` so that consumers stop holding. It contains NO reducer logic; when the arbiter is alive
    it does nothing.
    """
    cfg = config or client_config()
    ms = int(cfg["pendingMs"]) + ARBITER_MISSING_MARGIN_MS
    failed = json.dumps({"phase": "failed", "epoch": None, "attempt": 0, "reason": "arbiter_missing", "payload": None,
                         "hadOwnedParams": None})
    return ("<script>(function(){setTimeout(function(){if(window.__vvArbiterAlive)return;var n=0,t=setInterval(function(){"
            "n++;var dc=window.dash_clientside;if(dc&&dc.set_props){clearInterval(t);"
            f"dc.set_props('url-init',{{data:{failed}}});}}else if(n>100){{clearInterval(t);}}}},200);}},{ms});}})();</script>")
