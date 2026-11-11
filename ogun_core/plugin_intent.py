"""Loosely worded plugin requests: build vs. update. The logic lives in the shared kit (ogun_core.kit.plugin_intent)."""
from ogun_core.kit.plugin_intent import (  # noqa: F401 - re-exported
    installed_plugins,
    match_plugin,
    resolve_plugin_request,
)
