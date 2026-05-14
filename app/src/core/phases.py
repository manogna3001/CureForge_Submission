from time import perf_counter

from app.src.utils.metrics import get_metrics


PHASES = (
    "research",
    "hypothesize",
    "test",
    "synthesize",
)


PHASE_TRANSITIONS = {
    "research": ("hypothesize",),
    "hypothesize": ("research", "test"),
    "test": ("research", "synthesize"),
    "synthesize": ("research",),
}


_phase_timers: dict[str, float] = {}


def is_valid_transition(current_phase: str, next_phase: str) -> bool:
    """Validate allowed phase transitions, including fallback loops."""
    return next_phase in PHASE_TRANSITIONS.get(current_phase, ())


def record_phase_start(agent_id: str, phase: str) -> None:
    _phase_timers[f"{agent_id}:{phase}"] = perf_counter()


def record_phase_transition(agent_id: str, from_phase: str, to_phase: str) -> None:
    metrics = get_metrics()
    metrics.increment(
        "phase_transitions_total",
        labels={"phase_from": from_phase, "phase_to": to_phase},
    )
    start = _phase_timers.pop(f"{agent_id}:{from_phase}", None)
    if start is not None:
        metrics.histogram(
            "phase_duration_seconds",
            labels={"phase": from_phase},
            value=perf_counter() - start,
        )
