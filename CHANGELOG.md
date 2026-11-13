# Changelog

## Unreleased

- Desktop window: `Ogun.vbs` (or `--ui`) runs Ogun in its own app window with no terminal: chat with live tool activity, plugin list and builds, GPU controls, and Quit. Closing the window quits Ogun after any running task finishes.
- Builds have no attempt limit: plugin builds, tool builds and tool updates retry until they pass. `Ctrl+C` stops one; `OGUN_BUILD_MAX_ATTEMPTS` caps them.
- Failure research: on repeated failures the model writes a root-cause diagnosis before the next fix. Optional web search for the error (`OGUN_WEB_RESEARCH=1`, off by default).
- After five identical failures a build starts over from scratch.
- Fix feedback is trimmed to fit the context window, so long builds no longer lose the start of the prompt.
- Better test-versus-tool blame: crashes inside test code, separator-only mismatches and tests that expect Python type names over the requested words are treated as test defects.
- Fewer false rules from single words ("header row", "total", "group", "dotted").
- New output check for tools that report JSON value types (catches `true` reported as a number).
- Test fixtures are written byte-exact (no Windows line-ending changes), and helpers accept content-only calls.
- Plugins: `ToolResult` accepts `result=`/`error=`/`success=` and the `ToolResult.success(...)` / `ToolResult.error(...)` forms.
- `--build-plugin` follows the same no-limit setting as chat builds.
- Plugin builds: each fix prompt carries the request, the current tool and test, and the test lines that failed; history resets per attempt; five identical failures start the plugin over; every failed attempt prints its reason.
- Plugin tests must try every example the request quotes, and assertion messages must name the input.
- A failed plugin's test is kept in `plugins/quarantine` next to the failed tool.

## v0.1.0-beta (2026-11-03)

First public beta.

- Local chat and coding assistant running on Ollama (default model `qwen2.5-coder:14b`).
- Self-building plugins: writes a tool and a unit test, runs the test in isolation, mounts the tool only if it passes.
- Standalone command-line tool builder: describe a tool in chat and Ogun writes it, its tests and a README into `workspace/`, then checks it with smoke tests, output oracles, a 50 MB memory test and its own unit tests.
- Automatic hardware tuning and `/vram` GPU memory profiles.
- Local fine-tuning pipeline (data preparation tested; the fine-tune step has not yet been run end to end).
- Licensed under the Functional Source License 1.1 (MIT future licence).
