"""Data profiles: dataset-specific contracts (fields, features, reason codes) served by one platform.

``FRAUDSHIELD_PROFILE`` selects one: ``creditcard`` (ULB, default) or ``ieee`` (IEEE-CIS v2).
"""

from __future__ import annotations

from typing import Any

from fraudshield.profiles.base import ServingProfile


def get_profile(settings: Any) -> ServingProfile:
    if settings.profile == "ieee":
        from fraudshield.profiles.ieee_serving import IEEEProfile

        return IEEEProfile(settings.replay_file, settings.warmup_file)
    from fraudshield.profiles.creditcard import CreditCardProfile

    return CreditCardProfile(settings.replay_file)
