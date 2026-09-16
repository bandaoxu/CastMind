from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Callable, Dict, Optional

from pydantic_ai import Agent, RunContext  # type: ignore

from castmind.config import DatasetConfig, ExperimentConfig
from .prompts import get_agent_instructions


INVESTIGATOR_AGENT_PROMPT_FALLBACK = dedent(
    """
    You are InvestigatorAgent (paper §3.4.1). For each request:
      1. Call `gather_forecast_inputs` exactly once with dataset_name, window_offset,
         optional forecast_horizon, and optional reflective_feedback.
      2. The tool returns features F, F_selected (Eq. 6), knowledge K, and context E.
         Do NOT retrieve case-library clusters or neighbors (Generator owns that).
      3. Reply with a short confirmation that F_selected was obtained (list the names).
    """
)


def create_investigator_agent(
    model_name: str,
    cfg: ExperimentConfig | None,
    dataset_lookup: Dict[str, DatasetConfig],
    briefing_lookup: Dict[str, str],
    knowledge_lookup: Dict[str, str],
    prepare_investor_packet: Callable[..., dict],
    json_default: Callable[[Any], Any],
) -> Agent:
    instructions = get_agent_instructions("InvestigatorAgent", INVESTIGATOR_AGENT_PROMPT_FALLBACK)
    investigator_agent = Agent(model_name, instructions=instructions)
    globals()["RunContext"] = RunContext

    last_packet: dict[str, Any] = {}

    @investigator_agent.tool
    def gather_forecast_inputs(
        ctx: RunContext[None],
        dataset_name: str,
        window_offset: int = 0,
        forecast_horizon: Optional[int] = None,
        reflective_feedback: Optional[str] = None,
    ) -> dict:
        ds_cfg = dataset_lookup.get(dataset_name)
        if ds_cfg is None:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        try:
            window_offset_int = int(window_offset or 0)
        except Exception:
            window_offset_int = 0
        if forecast_horizon is not None:
            try:
                forecast_horizon = int(forecast_horizon)
            except Exception:
                forecast_horizon = None
        packet = prepare_investor_packet(
            cfg,
            ds_cfg,
            briefing_lookup,
            window_offset_int,
            forecast_horizon,
            reflective_feedback=reflective_feedback,
            knowledge_lookup=knowledge_lookup,
            include_case_evidence=False,
        )
        last_packet.clear()
        last_packet.update(packet)
        return packet

    # Expose last tool packet for Generator merge without re-parsing LLM text.
    investigator_agent._castmind_last_packet = last_packet  # type: ignore[attr-defined]
    return investigator_agent


__all__ = ["create_investigator_agent", "INVESTIGATOR_AGENT_PROMPT_FALLBACK"]
