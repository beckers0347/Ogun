"""Shared, app-neutral runtime kit (vendored into each app by sync_shared_kit.py; edit the canonical copy only).

Modules:
  gpu_lease      pinned Ollama load options + a cross-process model lease
  plugin_intent  build-vs-update interpretation of loosely worded plugin requests
  build_kit      staged plugin-builder helpers (test normalization, retry policy, disputed-case handling, shape check)
"""
KIT_VERSION = "1.0.0"
