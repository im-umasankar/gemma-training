"""FinQA data preprocessing transform for Kauldron.

Imported inside konfig.imports() in config_gemma4.py so it is treated
as a konfig-registered class and can be used in the transforms list.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import grain.python as grain


@dataclasses.dataclass(frozen=True)
class FinQAFormat(grain.MapTransform):
  """Formats TheFinAI/Fino1_Reasoning_Path_FinQA fields into prompt/response.

  Input fields:
    - "Open-ended Verifiable Question" → question text
    - "Complex_CoT"                   → chain-of-thought reasoning
    - "Response"                      → expected answer

  Output fields (consumed by gm.data.Seq2SeqTask):
    - "prompt":   formatted question + CoT
    - "response": answer text
  """

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    question = element.get("Open-ended Verifiable Question", "")
    cot = element.get("Complex_CoT", "")
    response = element.get("Response", "")
    prompt = (
        f"Question: {question}\n\n"
        f"Think step by step:\n{cot}"
    )
    return {"prompt": prompt, "response": response}
