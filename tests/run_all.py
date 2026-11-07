"""Master offline test runner for Ogun.

Executes all Ogun offline test suites:
- test_ogun_core.py
- test_training_data.py
- test_training_trainer.py

Prints aggregate summary and exits non-zero if any test suite fails.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_suite(name: str, script_path: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    root = script_path.parents[1]
    env["PYTHONPATH"] = f"{root};{root / 'training'}"

    t0 = time.time()
    res = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.time() - t0
    output = f"[{'PASS' if res.returncode == 0 else 'FAIL'}] {name} ({elapsed:.2f}s)\n{res.stdout}\n{res.stderr}"
    return res.returncode == 0, output


def main() -> int:
    tests_dir = Path(__file__).resolve().parent
    suites = [
        ("Ogun Core Test Suite", tests_dir / "test_ogun_core.py"),
        ("Ogun Training Data Pipeline Suite", tests_dir / "test_training_data.py"),
        ("Ogun Unsloth Trainer Suite", tests_dir / "test_training_trainer.py"),
        ("Ogun Hardware Auto-Tune Suite", tests_dir / "test_hardware.py"),
        ("Ogun Dev-Ability Fixes Suite", tests_dir / "test_dev_fixes.py"),
        ("Ogun Speed Optimizations Suite", tests_dir / "test_speed_optimizations.py"),
        ("Ogun Build & Upgrade Fixes Suite", tests_dir / "test_build_upgrade_fixes.py"),
        ("Ogun Option Behaviour Checks Suite", tests_dir / "test_option_checks.py"),
        ("Ogun Unlimited Builds Suite", tests_dir / "test_unlimited_builds.py"),
    ]

    print("=" * 70)
    print("                  OGUN MASTER TEST RUNNER")
    print("=" * 70)

    all_passed = True
    results = []

    for name, path in suites:
        print(f"[*] Running {name} ({path.name})...")
        passed, out = run_suite(name, path)
        results.append((name, path.name, passed, out))
        if not passed:
            all_passed = False
            print(f"[-] {name} FAILED!")
        else:
            print(f"[+] {name} PASSED.")

    print("\n" + "=" * 70)
    print("                       SUMMARY REPORT")
    print("=" * 70)
    for name, filename, passed, _ in results:
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str:8s} {name:38s} ({filename})")
    print("=" * 70)

    if not all_passed:
        print("\nDetails of failed test suites:")
        for name, filename, passed, out in results:
            if not passed:
                print(f"\n--- {name} ({filename}) Output ---")
                print(out)
        return 1

    print("\nAll Ogun offline test suites passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
