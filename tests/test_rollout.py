"""Offline tests for the framework-agnostic memory rollout state machine."""

from __future__ import annotations

from supersede.rollout import MemoryRollout


def test_initial_prompt_is_first_session_memory_prompt():
    ro = MemoryRollout(sessions=["S0", "S1"], question="Q?", budget=100)
    assert ro.phase == "memory"
    p = ro.current_prompt()
    assert "S0" in p and "NEW SESSION" in p
    assert "max 100 chars" in p


def test_steps_advance_through_sessions_then_question():
    ro = MemoryRollout(sessions=["S0", "S1", "S2"], question="Where now?", budget=100)
    p1 = ro.step("notes after S0")          # -> memory prompt for S1
    assert "S1" in p1 and ro.sidx == 1
    p2 = ro.step("notes after S1")          # -> memory prompt for S2
    assert "S2" in p2 and ro.sidx == 2
    p3 = ro.step("notes after S2")          # -> question prompt
    assert ro.phase == "answer"
    assert "Where now?" in p3 and "NOTES:" in p3


def test_answer_step_completes_and_stores_answer():
    ro = MemoryRollout(sessions=["S0"], question="Q?", budget=100)
    q = ro.step("my notes")                 # -> question prompt
    assert ro.phase == "answer" and "Q?" in q
    nxt = ro.step("the final answer")
    assert nxt is None
    assert ro.done and ro.final_answer == "the final answer"


def test_full_protocol_turn_count_is_sessions_plus_one():
    sessions = [f"S{i}" for i in range(5)]
    ro = MemoryRollout(sessions=sessions, question="Q?", budget=50)
    turns = 0
    # simulate a stub model that just emits placeholder text each turn
    ro.current_prompt()  # exercise the initial prompt path
    while True:
        turns += 1
        nxt = ro.step(f"reply {turns}")
        if nxt is None:
            break
    assert turns == len(sessions) + 1     # 5 memory writes + 1 answer
    assert ro.final_answer == f"reply {turns}"


def test_notes_are_capped_to_budget():
    ro = MemoryRollout(sessions=["S0", "S1"], question="Q?", budget=10)
    ro.step("x" * 500)
    assert len(ro.notes) == 10


def test_system_prompt_switches_for_answer_phase():
    ro = MemoryRollout(sessions=["S0"], question="Q?", budget=100)
    assert "NOTES field" in ro.system_prompt()      # memory phase
    ro.step("notes")
    assert "answer the user's question".lower() in ro.system_prompt().lower()
