#!/usr/bin/env python3
"""CPU-safe diagnostics for the legacy-equivalent ST-Spec StepPlan scaffold."""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import types
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# This diagnostic is intentionally CPU-safe. The public nano_pearl package
# imports GPU model modules from __init__, so create lightweight package stubs
# before importing the scheduler-only modules used by this test.
nano_pkg = types.ModuleType("nano_pearl")
nano_pkg.__path__ = [os.path.join(REPO_ROOT, "nano_pearl")]
sys.modules.setdefault("nano_pearl", nano_pkg)
engine_pkg = types.ModuleType("nano_pearl.pearl_engine")
engine_pkg.__path__ = [os.path.join(REPO_ROOT, "nano_pearl", "pearl_engine")]
sys.modules.setdefault("nano_pearl.pearl_engine", engine_pkg)

Scheduler = importlib.import_module("nano_pearl.pearl_engine.scheduler").Scheduler
Sequence = importlib.import_module("nano_pearl.pearl_engine.sequence").Sequence
stspec_plan = importlib.import_module("nano_pearl.pearl_engine.stspec_plan")
PlanRole = stspec_plan.PlanRole


def make_scheduler() -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        eos=-1,
        num_kvcache_blocks=16,
        kvcache_block_size=256,
    )
    return Scheduler(config)


def add_dummy_requests(scheduler: Scheduler) -> list[Sequence]:
    seqs = [
        Sequence(
            [1, 2, 3],
            request_id="req-a",
            slo_class="tight",
            slo_tpot_ms=25.0,
            per_request_gamma=2,
        ),
        Sequence(
            [4, 5],
            request_id="req-b",
            slo_class="relaxed",
            slo_tpot_ms=100.0,
            per_request_gamma=8,
        ),
    ]
    for seq in seqs:
        assert seq.home_batch_id is None
        scheduler.add(seq)
    assert [seq.home_batch_id for seq in seqs] == [0, 1]
    return seqs


def assert_legacy_plan_fields(step_plan, seqs, *, is_prefill: bool, expected_budget: int) -> None:
    seq_ids = [seq.seq_id for seq in seqs]
    assert step_plan.legacy_equivalent is True
    assert step_plan.scheduled_seq_ids == seq_ids
    assert step_plan.eager_seq_ids == []
    assert all(request.is_eager is False for request in step_plan.requests)
    expected_home_batch_ids = {seq.seq_id: idx % 2 for idx, seq in enumerate(seqs)}
    assert {request.seq_id: request.home_batch_id for request in step_plan.requests} == expected_home_batch_ids
    assert all(request.eager_budget == 0 for request in step_plan.requests)
    assert all(request.draft_budget == expected_budget for request in step_plan.requests)
    assert all(request.effective_gamma == 4 for request in step_plan.requests)
    if is_prefill:
        assert all(request.role == PlanRole.PREFILL for request in step_plan.requests)
    else:
        assert all(request.role == PlanRole.DRAFT for request in step_plan.requests)
        assert step_plan.draft_home_seq_ids == seq_ids
    assert step_plan.is_eager_per_seq == {seq.seq_id: False for seq in seqs}
    assert step_plan.home_batch_id_per_seq == expected_home_batch_ids
    assert step_plan.effective_gamma_per_seq == {seq.seq_id: 4 for seq in seqs}
    signature = step_plan.signature()
    assert signature["plan_id"] == step_plan.plan_id
    assert signature["legacy_equivalent"] is True
    assert signature["scheduled_seq_ids"] == seq_ids
    assert signature["request_ids"] == [seq.request_id for seq in seqs]
    assert signature["effective_gamma_per_seq"] == {str(seq.seq_id): 4 for seq in seqs}
    assert signature["home_batch_id_per_seq"] == {str(seq_id): home_batch_id for seq_id, home_batch_id in expected_home_batch_ids.items()}
    assert signature["is_eager_per_seq"] == {str(seq.seq_id): False for seq in seqs}
    assert stspec_plan.step_plan_digest(step_plan) == step_plan.digest()


def main() -> None:
    legacy_scheduler = make_scheduler()
    planned_scheduler = make_scheduler()
    legacy_seqs = add_dummy_requests(legacy_scheduler)
    planned_seqs = add_dummy_requests(planned_scheduler)

    legacy_prefill, legacy_is_prefill = legacy_scheduler.schedule()
    planned_prefill, planned_is_prefill, prefill_plan = planned_scheduler.schedule_with_plan(
        runner_role="draft_prefill",
        execution_mode="parallel_pearl",
        decode_ready_mode=False,
        default_gamma=4,
    )

    assert [seq.home_batch_id for seq in planned_seqs] == [0, 1]
    assert [seq.service_metadata()["home_batch_id"] for seq in planned_seqs] == [0, 1]
    assert [pickle.loads(pickle.dumps(seq)).home_batch_id for seq in planned_seqs] == [0, 1]

    assert [seq.request_id for seq in planned_prefill] == [seq.request_id for seq in legacy_prefill]
    assert planned_is_prefill == legacy_is_prefill == True
    assert planned_prefill == planned_seqs
    assert_legacy_plan_fields(prefill_plan, planned_prefill, is_prefill=True, expected_budget=1)

    legacy_decode, legacy_decode_is_prefill = legacy_scheduler.schedule()
    planned_decode, planned_decode_is_prefill, decode_plan = planned_scheduler.schedule_with_plan(
        runner_role="draft",
        execution_mode="parallel_pearl",
        decode_ready_mode=False,
        default_gamma=4,
    )

    assert [seq.request_id for seq in planned_decode] == [seq.request_id for seq in legacy_decode]
    assert planned_decode_is_prefill == legacy_decode_is_prefill == False
    assert planned_decode == planned_seqs
    assert_legacy_plan_fields(decode_plan, planned_decode, is_prefill=False, expected_budget=4)

    ar_scheduler = make_scheduler()
    add_dummy_requests(ar_scheduler)
    ar_scheduler.schedule()
    ar_decode, ar_is_prefill, ar_plan = ar_scheduler.schedule_with_plan(
        runner_role="verify",
        execution_mode="ar",
        decode_ready_mode=False,
        default_gamma=4,
    )
    assert ar_is_prefill is False
    assert all(request.role == PlanRole.TARGET for request in ar_plan.requests)
    assert all(request.draft_budget == 1 for request in ar_plan.requests)
    assert ar_plan.target_seq_ids == [seq.seq_id for seq in ar_decode]

    print("ST-Spec StepPlan scaffold diagnostics passed")


if __name__ == "__main__":
    main()
