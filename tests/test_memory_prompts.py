"""Tests for memory prompt guardrails."""

from __future__ import annotations

from heta.mem.prompts import BATCH_CONFLICT_JUDGE_PROMPT, EPISODE_DEDUP_PROMPT



def test_batch_conflict_prompt_is_conservative_about_related_facts() -> None:
    prompt = BATCH_CONFLICT_JUDGE_PROMPT

    assert "different attributes/aspects" in prompt
    assert "adds detail" in prompt
    assert "both facts can be true" in prompt
    assert "When in doubt, keep both" in prompt
    assert "Zhang Ming got a raise" in prompt
    assert "Zhang Ming moved jobs to Zhipu" in prompt


def test_episode_dedup_prompt_is_conservative_about_similar_events() -> None:
    prompt = EPISODE_DEDUP_PROMPT

    assert "same concrete real-world event" in prompt
    assert "same main participants" in prompt
    assert "Do NOT mark duplicate merely because" in prompt
    assert "different granularity or a later update" in prompt
    assert "When in doubt, keep both" in prompt
