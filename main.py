"""Ogun standalone autonomous coding assistant CLI entrypoint."""
from __future__ import annotations

import argparse
import atexit
import os
import re
import sys

from ogun_core import autotune
from ogun_core.config import BUILD_MAX_ATTEMPTS, MODEL_NAME, OLLAMA_HOST, PLUGIN_BUILDER_PROMPT
from ogun_core.config import set_vram_profile
from ogun_core.engine import CoreEngine, OllamaClient, handle_vram_command, unload_model
from ogun_core.paths import init_user_workspace
from ogun_core.plugin_builder import build_plugin_staged, parse_code_blocks
from ogun_core.plugin_manager import PluginManager


UNLOAD_PHRASES = {
    "/unload", "unload", "unload model", "unload the model", "free gpu", "free the gpu", "free up gpu",
    "free vram", "free up vram", "release gpu memory", "release vram",
}


def is_unload_command(text: str) -> bool:
    """True only for an exact unload phrase, so ordinary sentences ('unload the dishwasher') are never hijacked."""
    return " ".join((text or "").lower().replace("?", " ").replace(".", " ").split()) in UNLOAD_PHRASES


def main() -> None:
    parser = argparse.ArgumentParser(description="Ogun Autonomous Coding Assistant")
    parser.add_argument("--once", type=str, help="Execute a single turn and exit.")
    parser.add_argument("--build-plugin", type=str, help="Prompt the model to build, verify, and mount a plugin.")
    parser.add_argument("--model", type=str, default=None, help=f"Ollama model to use (default: {MODEL_NAME}).")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool execution.")
    parser.add_argument("--unload", action="store_true",
                        help="Unload the model from Ollama (free its GPU memory) and exit.")
    parser.add_argument("--vram", type=str, default=None, metavar="PROFILE",
                        help="Start with a fixed VRAM profile: max | shared | cpu (skips hardware auto-tuning). Change it in chat with /vram.")
    autotune.add_arguments(parser)

    args = parser.parse_args()
    args.model_explicit = args.model            # only a model typed on the command line is exempt from the 'does it fit' check
    args.model = args.model or MODEL_NAME

    # --unload must work even when nothing else can: handle it before any connection check or workspace setup.
    if args.unload:
        result = unload_model(model=args.model)
        print(f"[{'OK' if result['unloaded'] else 'WARN'}] {result['detail']}")
        sys.exit(0)

    if args.hardware_info:
        sys.exit(autotune.hardware_info(args, args.model, OLLAMA_HOST))

    if args.vram:
        ok, _changed, profile, num_gpu = set_vram_profile(args.vram)
        if not ok:
            print(f"[ERROR] Unknown --vram profile '{args.vram}'. Use max, shared or cpu.", file=sys.stderr)
            sys.exit(2)
        print(f"[*] VRAM profile: {profile} (num_gpu={num_gpu})")

    # Size the runtime to this machine (GPU layers, context window, threads, and a model that fits). Never fatal.
    tuned = autotune.boot(args, args.model, OLLAMA_HOST)
    if tuned and tuned.get("model_tag") and tuned["model_tag"] != args.model:
        args.model = tuned["model_tag"]

    # Pre-flight check on Ollama connection
    client = OllamaClient(model=args.model)
    connected, conn_msg = client.check_connection()
    if not connected:
        print(f"[ERROR] {conn_msg}", file=sys.stderr)
        sys.exit(1)

    # Free the GPU when Ogun exits (normal exit, Ctrl+C, --once, --build-plugin). Opt out: OGUN_UNLOAD_ON_EXIT=0.
    if os.environ.get("OGUN_UNLOAD_ON_EXIT", "1") != "0":
        _unloaded = {"done": False}

        def _unload_on_exit() -> None:
            if not _unloaded["done"]:
                _unloaded["done"] = True
                unload_model(model=args.model)

        atexit.register(_unload_on_exit)

    # Initialize workspace directories on boot
    init_user_workspace()

    plugin_manager = PluginManager()
    plugin_manager.discover_tools()

    engine = CoreEngine(
        client=client,
        plugin_manager=None if args.no_tools else plugin_manager,
    )

    # 1. Build plugin mode (staged with self-correction)
    if args.build_plugin:
        print(f"[*] Building plugin for request: {args.build_plugin}...")
        success, msg, _plugin_name = build_plugin_staged(
            engine=engine,
            plugin_manager=plugin_manager,
            request_text=args.build_plugin,
            max_attempts=BUILD_MAX_ATTEMPTS,
        )
        if success:
            print(msg)
            sys.exit(0)
        else:
            print(msg, file=sys.stderr)
            sys.exit(1)

    # 2. Single turn mode
    if args.once:
        reply = engine.run_turn(args.once)
        print(reply)
        sys.exit(0)

    # 3. Interactive CLI loop
    print("=" * 60)
    print(f"  Ogun Autonomous Coding Assistant (Model: {args.model})")
    print("  Type 'exit' or 'quit' to close. Press Ctrl+C to cancel a turn.")
    print("  GPU memory: /vram auto | /vram shared | /vram max | /vram cpu | /vram free | /vram status")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nOgun > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            vram_reply = handle_vram_command(user_input, model=args.model)
            if vram_reply is not None:
                print(f"\n{vram_reply}")
                continue
            if is_unload_command(user_input):
                result = unload_model(model=args.model)
                print(f"\n{result['detail']} The next question reloads the model (about 40 seconds).")
                continue
            _gpu_note = autotune.refresh(OLLAMA_HOST, args.model)
            if _gpu_note:
                print(f"[Hardware] {_gpu_note}")
            reply = engine.run_turn(user_input)
            print(f"\n{reply}")
        except KeyboardInterrupt:
            print("\nTurn canceled by user.")
        except EOFError:
            break


if __name__ == "__main__":
    main()
