"""Simple RL Research
====================

Modular research framework for training reasoning-capable LLMs with
REINFORCE.  Public API is intentionally small – most components are
accessed via their sub-packages (see ``generation``, ``reward``,
``trainer``).  Importing the top-level package mainly guarantees that all
sub-modules are discoverable by the python import machinery.
"""

__all__ = [
    "generation",
    "reward",
    "trainer",
    "utils",
]