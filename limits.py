"""
Single authoritative source for the UI limits that more than one module needs.

Standard library only (a leaf module: it imports nothing from VibroVis), so the layout, the callbacks, the pure
``url_state`` module and the tests can all read the same constants without import cycles.

* ``MAX_POINTS_LIMIT``  hard upper bound of the "Max points" control (UI ``max=`` and server-side clamp).
                        Override with the environment variable ``VIBROVIS_MAX_POINTS_LIMIT``.
* ``DEFAULT_MAX_POINTS`` the control's default value (unchanged: 10,000).
* ``HOUR_SLIDER_STEP``  step of the hour range slider; URL restoration snaps hours to it.
* ``url_init_config()`` deadlines of the URL-initialisation state machine (see ``assets/url_restore.js``), validated.
"""
import math
import os


def _env_positive_int(name, default):
    try:
        value = int(float(os.environ[name]))
        return value if value > 0 else default
    except (KeyError, ValueError):
        return default


MAX_POINTS_LIMIT = _env_positive_int("VIBROVIS_MAX_POINTS_LIMIT", 100_000)
DEFAULT_MAX_POINTS = 10_000
HOUR_SLIDER_STEP = 0.5


# ---- URL initialisation deadlines ---------------------------------------------------------------------------------------
# pending: how long initialisation may take to START; running: how long a started initialisation may take to answer;
# grace: how long after the server callback ENDED a missing answer is waited for; tick: deadline-check period.
URL_INIT_DEFAULTS_S = {"pending": 10.0, "running": 60.0, "grace": 1.5, "tick": 0.5}
URL_INIT_BOUNDS_S = {"pending": (1.0, 300.0), "running": (1.0, 3600.0), "grace": (0.2, 30.0), "tick": (0.1, 5.0)}
URL_INIT_ENV = {"pending": "VIBROVIS_URL_INIT_PENDING_S", "running": "VIBROVIS_URL_INIT_RUNNING_S",
                "grace": "VIBROVIS_URL_INIT_GRACE_S", "tick": "VIBROVIS_URL_INIT_TICK_S"}


def _env_seconds(environ, name, default, lo, hi):
    """Validated float from the environment; anything missing, non-numeric, non-finite or out of range -> default."""
    raw = environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default, False
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default, True
    if not math.isfinite(value) or not (lo <= value <= hi):
        return default, True
    return value, False


def url_init_config(environ=None):
    """
    Deadlines for the URL-initialisation state machine, in milliseconds: ``{pendingMs, runningMs, graceMs, tickMs,
    invalid}`` where ``invalid`` lists the environment variables that were ignored (so callers can log them).

    Every value is validated and falls back to its safe default when invalid. Combinations must also make sense:
    ``running >= pending`` and ``grace >= 2 * tick``; otherwise the affected pair falls back to its defaults.
    """
    environ = os.environ if environ is None else environ
    vals, invalid = {}, []
    for key, default in URL_INIT_DEFAULTS_S.items():
        lo, hi = URL_INIT_BOUNDS_S[key]
        vals[key], bad = _env_seconds(environ, URL_INIT_ENV[key], default, lo, hi)
        if bad:
            invalid.append(URL_INIT_ENV[key])
    if vals["running"] < vals["pending"]:
        invalid += [URL_INIT_ENV["pending"], URL_INIT_ENV["running"]]
        vals["pending"], vals["running"] = URL_INIT_DEFAULTS_S["pending"], URL_INIT_DEFAULTS_S["running"]
    if vals["grace"] < 2 * vals["tick"]:
        invalid += [URL_INIT_ENV["grace"], URL_INIT_ENV["tick"]]
        vals["grace"], vals["tick"] = URL_INIT_DEFAULTS_S["grace"], URL_INIT_DEFAULTS_S["tick"]
    return {"pendingMs": int(round(vals["pending"] * 1000)), "runningMs": int(round(vals["running"] * 1000)),
            "graceMs": int(round(vals["grace"] * 1000)), "tickMs": int(round(vals["tick"] * 1000)),
            "invalid": sorted(set(invalid))}


URL_RESTORE_TIMEOUT_ENV = "VIBROVIS_URL_RESTORE_TIMEOUT"       # seconds; watchdog for one whole URL restoration
URL_RESTORE_TIMEOUT_DEFAULT_S = 180.0
URL_RESTORE_TIMEOUT_BOUNDS_S = (10.0, 3600.0)


def url_restore_timeout_ms(environ=None):
    """``(milliseconds, invalid_env_names)`` for the restoration watchdog; invalid values fall back to the default."""
    environ = os.environ if environ is None else environ
    lo, hi = URL_RESTORE_TIMEOUT_BOUNDS_S
    value, bad = _env_seconds(environ, URL_RESTORE_TIMEOUT_ENV, URL_RESTORE_TIMEOUT_DEFAULT_S, lo, hi)
    return int(round(value * 1000)), ([URL_RESTORE_TIMEOUT_ENV] if bad else [])
