"""Test defaults: the app's own GPU tuning is under test unless a test turns the shared lock on itself.

Leases stay in the real shared folder on purpose: a test run's exit-time unload must see another app's live job and
leave the model loaded. Tests that write leases point KIT_LEASE_DIR at their own tmp_path."""
import os

os.environ.setdefault("KIT_GPU_LOCK", "0")
