"""Smoke test for the local W&B server.

Checks env + connectivity, then logs a tiny run end-to-end. Auth comes from
WANDB_API_KEY / WANDB_BASE_URL in the environment — no interactive `wandb login`.
Run from a shell that has sourced ~/.bash_aliases:

    source ~/.bash_aliases
    source /home/sonofr/python_venvs/.venv/bin/activate
    python wandb-local/check_wandb.py
"""

import os
import sys
import urllib.request

BASE_URL = os.environ.get("WANDB_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("WANDB_API_KEY", "")


def preflight():
    """Cheap checks before touching wandb. Returns True if safe to proceed."""
    ok = True

    print(f"WANDB_BASE_URL = {BASE_URL or '(unset)'}")
    if not BASE_URL:
        print("  ✗ export WANDB_BASE_URL=http://wandb-local.localhost:8421")
        ok = False

    n = len(API_KEY)
    print(f"WANDB_API_KEY  = {'set' if API_KEY else '(unset)'}, length {n}")
    if n == 0:
        print("  ✗ empty — WANDB_LOCAL_TOKEN didn't resolve")
        ok = False
    elif n != 40:
        # A W&B API key is exactly 40 chars; this is almost certainly the wrong
        # value (the license/access token, not the key). The server will reject
        # it with "API key must be 40 characters long".
        print(f"  ⚠ a W&B API key is 40 chars, not {n}. WANDB_LOCAL_TOKEN likely holds")
        print(f"    the wrong value. Copy the real key from {BASE_URL}/authorize")
        print("    (or avatar → User settings → API keys) into WANDB_LOCAL_TOKEN.")

    if BASE_URL:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/healthz", timeout=5) as r:
                print(f"server /healthz = HTTP {r.status}")
        except Exception as e:  # noqa: BLE001 - report any connectivity failure
            print(f"  ✗ server not reachable: {e}")
            ok = False

    return ok


def smoke_run():
    """Log a tiny run; raises on auth/connection failure."""
    import wandb

    print(f"wandb {wandb.__version__}")
    run = wandb.init(project="smoke-test", config={"lr": 1e-3, "epochs": 10})
    for step in range(10):
        run.log({"loss": 1.0 / (step + 1), "acc": step / 10.0}, step=step)
    url = run.get_url()
    run.finish()
    print(f"✓ logged run: {url}")


if __name__ == "__main__":
    if not preflight():
        print("\npreflight failed — fix the above, then re-run.")
        sys.exit(1)
    try:
        smoke_run()
    except ImportError:
        print("\n✗ wandb not installed: pip install wandb")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - surface the real server error clearly
        msg = str(e)
        print(f"\n✗ run failed: {type(e).__name__}: {msg}")
        if "40 char" in msg or "malformed" in msg.lower():
            print(f"  → key is wrong; grab the 40-char key from {BASE_URL}/authorize")
        sys.exit(1)
    print("\nAll good — open the run URL above.")
