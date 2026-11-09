"""Training subsystem: online loop, trainer, round scheduling, and loss dispatch.

Kept import-light (no eager torch import at package load): import the
submodules (``training.trainer`` / ``training.objective`` / ``training.online``)
directly when you need them."""
