"""Legacy-equivalent ST-Spec step planning scaffold.

This module intentionally does *not* implement ST-Spec scheduling features yet.
It records the current Scheduler.schedule() decision in a structured StepPlan so
future work can add dual batches, eager paths, and SLO-aware allocation behind a
stable interface without changing today's runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nano_pearl.pearl_engine.sequence import Sequence


class PlanRole(str, Enum):
    """Logical role assigned to a request inside a StepPlan."""

    PREFILL = "prefill"
    DRAFT = "draft"
    TARGET = "target"


@dataclass(frozen=True)
class PlanRequest:
    seq_id: int
    request_id: Any
    role: PlanRole
    home_batch_id: int | str | None
    is_eager: bool
    effective_gamma: int
    draft_budget: int
    eager_budget: int
    slo_class: str | None
    slo_tpot_ms: float | None
    per_request_gamma: int | None


@dataclass(frozen=True)
class StepPlan:
    plan_id: int
    execution_mode: str
    decode_ready_mode: bool
    runner_role: str
    is_prefill: bool
    legacy_equivalent: bool
    scheduled_seq_ids: list[int] = field(default_factory=list)
    target_seq_ids: list[int] = field(default_factory=list)
    draft_home_seq_ids: list[int] = field(default_factory=list)
    eager_seq_ids: list[int] = field(default_factory=list)
    requests: list[PlanRequest] = field(default_factory=list)

    @property
    def effective_gamma_per_seq(self) -> dict[int, int]:
        return {request.seq_id: request.effective_gamma for request in self.requests}

    @property
    def home_batch_id_per_seq(self) -> dict[int, int | str | None]:
        return {request.seq_id: request.home_batch_id for request in self.requests}

    @property
    def is_eager_per_seq(self) -> dict[int, bool]:
        return {request.seq_id: request.is_eager for request in self.requests}

    @property
    def request_ids(self) -> list[Any]:
        return [request.request_id for request in self.requests]

    def signature(self) -> dict[str, Any]:
        """Return a compact JSON-serializable plan signature.

        The signature is diagnostic-only: it summarizes the legacy-equivalent
        decision already made by Scheduler.schedule() and does not participate
        in scheduling, verification, or KV-cache behavior. Per-sequence maps use
        string keys so the signature is stable under JSON serialization.
        """
        return {
            "plan_id": self.plan_id,
            "execution_mode": self.execution_mode,
            "decode_ready_mode": self.decode_ready_mode,
            "runner_role": self.runner_role,
            "is_prefill": self.is_prefill,
            "legacy_equivalent": self.legacy_equivalent,
            "scheduled_seq_ids": list(self.scheduled_seq_ids),
            "request_ids": list(self.request_ids),
            "effective_gamma_per_seq": {
                str(seq_id): gamma
                for seq_id, gamma in self.effective_gamma_per_seq.items()
            },
            "home_batch_id_per_seq": {
                str(seq_id): home_batch_id
                for seq_id, home_batch_id in self.home_batch_id_per_seq.items()
            },
            "is_eager_per_seq": {
                str(seq_id): is_eager
                for seq_id, is_eager in self.is_eager_per_seq.items()
            },
        }

    def digest(self) -> str:
        return step_plan_digest(self)


def step_plan_signature(step_plan: StepPlan) -> dict[str, Any]:
    return step_plan.signature()


def step_plan_digest(step_plan: StepPlan) -> str:
    encoded = json.dumps(
        step_plan_signature(step_plan),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def role_from_runner(runner_role: str, is_prefill: bool) -> PlanRole:
    if is_prefill or runner_role.endswith("_prefill"):
        return PlanRole.PREFILL
    if "draft" in runner_role:
        return PlanRole.DRAFT
    return PlanRole.TARGET


def build_legacy_step_plan(
    *,
    plan_id: int,
    seqs: list[Sequence],
    is_prefill: bool,
    runner_role: str,
    execution_mode: str,
    decode_ready_mode: bool,
    default_gamma: int,
) -> StepPlan:
    """Wrap the existing scheduler output in a legacy-equivalent StepPlan.

    The supplied ``seqs`` are used as-is. No sequence ordering, batching,
    budgets, KV state, or verification layout is changed by this helper. V3A
    only shadows deterministic home-batch membership for future two-batch
    scheduling and does not act on that metadata.
    """
    scheduled_seq_ids = [seq.seq_id for seq in seqs]
    role = role_from_runner(runner_role, is_prefill)
    effective_gamma = max(int(default_gamma), 1)
    is_pearl_decode = (
        execution_mode in {"parallel_pearl", "serialized_pearl"}
        and not is_prefill
    )
    draft_budget = effective_gamma if is_pearl_decode else 1

    if "draft" in runner_role:
        draft_home_seq_ids = list(scheduled_seq_ids)
        target_seq_ids: list[int] = []
    else:
        draft_home_seq_ids = []
        target_seq_ids = list(scheduled_seq_ids)

    requests = [
        PlanRequest(
            seq_id=seq.seq_id,
            request_id=seq.request_id,
            role=role,
            home_batch_id=seq.home_batch_id,
            is_eager=False,
            effective_gamma=effective_gamma,
            draft_budget=draft_budget,
            eager_budget=0,
            slo_class=seq.slo_class,
            slo_tpot_ms=seq.slo_tpot_ms,
            per_request_gamma=seq.per_request_gamma,
        )
        for seq in seqs
    ]

    return StepPlan(
        plan_id=plan_id,
        execution_mode=execution_mode,
        decode_ready_mode=decode_ready_mode,
        runner_role=runner_role,
        is_prefill=is_prefill,
        legacy_equivalent=True,
        scheduled_seq_ids=scheduled_seq_ids,
        target_seq_ids=target_seq_ids,
        draft_home_seq_ids=draft_home_seq_ids,
        eager_seq_ids=[],
        requests=requests,
    )
