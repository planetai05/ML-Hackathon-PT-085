"""Validation of the API's GET /health body for the UI (no Streamlit import, so it is unit-testable).

The UI must never crash on an unexpected body, and must never mistake another service for the
PathSnap API. A typical cause: some other program (e.g. a Docker container) is listening on the
same host:port and answers /health with something like {"status": "ok"}.
"""
from __future__ import annotations

SERVICE_ID = "pathsnap-api"
MODES = ("real", "mock", "unavailable")

# optional fields the UI reads, with safe display defaults
DEFAULTS = {
    "status": "degraded",
    "model_loaded": False,
    "model_name": None,
    "artifact_version": None,
    "predictor": None,
    "predictor_error": None,
    "trips_dir": None,
    "trips_dir_exists": False,
    "n_trips": 0,
}


def fmt_value(v) -> str:
    """Display text for the UI details table: '–' if missing, 2 decimals for fractional numbers."""
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.2f}"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}"
    return str(v)


class HealthContractError(ValueError):
    """The /health body does not come from a compatible PathSnap API."""


def parse_health(body, url: str = "") -> dict:
    where = f" at {url}" if url else ""
    if not isinstance(body, dict):
        raise HealthContractError(f"The service{where} returned an unexpected /health body ({type(body).__name__}).")
    is_pathsnap = body.get("service") == SERVICE_ID or "predictor_mode" in body
    if not is_pathsnap:
        keys = ", ".join(sorted(map(str, body))) or "none"
        raise HealthContractError(
            f"The service{where} is not the PathSnap API: its /health response has no 'predictor_mode' "
            f"(fields: {keys}). Another program may be using this host/port – check the API URL or port.")
    mode = body.get("predictor_mode")
    if mode not in MODES:
        raise HealthContractError(f"The PathSnap API{where} reported an unknown predictor_mode {mode!r}; "
                                  f"expected one of {', '.join(MODES)}.")
    out = {**DEFAULTS, **{k: v for k, v in body.items() if v is not None or k not in DEFAULTS}}
    out["predictor_mode"] = mode
    try:
        out["n_trips"] = int(out["n_trips"] or 0)
    except (TypeError, ValueError):
        out["n_trips"] = 0
    return out
