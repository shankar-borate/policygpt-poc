"""Domain-specific Config overrides.

Each entry maps a domain_type string to a dict of Config field overrides that
are applied in Config.__post_init__ after all other preset/model scaling.
This keeps infrastructure sizing decisions (token budgets, retrieval limits)
separate from the LLM prompt and entity settings that live in core/domain/.

To add a new domain override:
  1. Add an entry below using the same domain_type key registered in core/domain/.
  2. Only list fields that differ from the base Config defaults.
  3. chat_max_output_tokens is a hard minimum — if the model floor (e.g. 120b = 6000)
     already exceeds this value, the higher value wins.
"""

from typing import Any


DOMAIN_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "policy": {
        # Enterprise policy answers are often lengthy — procedures, eligibility
        # tables, multi-step approval chains.  Set a high minimum so the model
        # has room to give complete, well-structured answers.
        "chat_max_output_tokens": 5400,
    },
    "contest": {
        # Contest answers are focused and concise — thresholds, reward amounts,
        # timelines.  A moderate budget is sufficient.
        "chat_max_output_tokens": 2700,
    },
}
