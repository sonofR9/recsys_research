import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_REAPER_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True)
class ServicePaths:
    root: Path

    @property
    def pending(self) -> Path:
        return self.root / "pending"

    @property
    def dispatched(self) -> Path:
        return self.root / "dispatched"

    @property
    def completed(self) -> Path:
        return self.root / "completed"

    @property
    def failed(self) -> Path:
        return self.root / "failed"

    @property
    def stable(self) -> Path:
        return self.root / "stable"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def acknowledgements(self) -> Path:
        return self.root / "acks"

    @property
    def batches(self) -> Path:
        return self.root / "batches"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def lock(self) -> Path:
        return self.root / "service.lock"

    @property
    def paused(self) -> Path:
        return self.root / "paused"

    @property
    def stop(self) -> Path:
        return self.root / "stop"

    @property
    def engine_ready(self) -> Path:
        return self.root / "engine.ready"

    @property
    def daemon_log(self) -> Path:
        return self.root / "service.log"

    def create(self) -> None:
        for directory in (
            self.pending,
            self.dispatched,
            self.completed,
            self.failed,
            self.stable,
            self.results,
            self.acknowledgements,
            self.batches,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _pid_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _process_start_time(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return None


def _is_service_process(pid: int, instance_token: str) -> bool:
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    decoded = {argument.decode(errors="replace") for argument in arguments}
    return (
        str(Path(__file__).resolve()) in decoded
        and "_serve" in decoded
        and instance_token in decoded
    )


def _service_lock_is_held(paths: ServicePaths) -> bool:
    lock_file = paths.lock.open("a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return True
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()
    return False


def _service_pid(paths: ServicePaths) -> int | None:
    try:
        status = _read_json(paths.status)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    pid = status.get("pid")
    instance_token = status.get("instance_token")
    start_time = status.get("pid_start_time")
    if (
        isinstance(pid, int)
        and isinstance(instance_token, str)
        and instance_token
        and isinstance(start_time, int)
        and _pid_is_running(pid)
        and _process_start_time(pid) == start_time
        and _is_service_process(pid, instance_token)
        and paths.engine_ready.exists()
        and _service_lock_is_held(paths)
    ):
        return pid
    return None


def _count(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.json"))


def _status_payload(
    paths: ServicePaths,
    *,
    pid: int,
    instance_token: str | None,
) -> dict[str, Any]:
    return {
        "running": True,
        "pid": pid,
        "pid_start_time": _process_start_time(pid),
        "instance_token": instance_token,
        "paused": paths.paused.exists(),
        "active": _count(paths.dispatched),
        "queued": _count(paths.pending),
        "completed": _count(paths.completed),
        "failed": _count(paths.failed),
        "updated_at": time.time(),
    }


def _write_status(paths: ServicePaths, instance_token: str) -> None:
    _atomic_json(
        paths.status,
        _status_payload(
            paths,
            pid=os.getpid(),
            instance_token=instance_token,
        ),
    )


def _fail_dispatched(paths: ServicePaths, reason: str) -> None:
    for path in paths.dispatched.glob("*.json"):
        job = _read_json(path)
        job["exit_code"] = None
        job["finished_at"] = time.time()
        job["failure"] = reason
        _atomic_json(paths.failed / path.name, job)
        path.unlink()


def _collect_results(paths: ServicePaths) -> None:
    for result in paths.results.glob("*.result"):
        job_id = result.stem
        job_path = paths.dispatched / f"{job_id}.json"
        if not job_path.exists():
            result.unlink()
            continue
        try:
            exit_code = int(result.read_text().strip())
        except ValueError:
            exit_code = 1
        job = _read_json(job_path)
        job["exit_code"] = exit_code
        job["finished_at"] = time.time()
        destination = paths.completed if exit_code == 0 else paths.failed
        _atomic_json(destination / job_path.name, job)
        job_path.unlink()
        result.unlink()


def _next_job(paths: ServicePaths) -> tuple[Path, dict[str, Any]] | None:
    try:
        pending = next(iter(sorted(paths.pending.glob("*.json"))))
    except StopIteration:
        return None
    job = _read_json(pending)
    dispatched = paths.dispatched / f"{job['id']}.json"
    os.replace(pending, dispatched)
    return dispatched, job


def _scheduler_command(job: dict[str, Any], paths: ServicePaths) -> str:
    arguments = [
        "_service_enqueue",
        job["script"],
        job["id"],
        str(paths.results),
        job["data_group"],
        job["run"],
        *job["environment"],
    ]
    return shlex.join(arguments) + "\n"


def _all_batches_sealed(paths: ServicePaths) -> bool:
    batch_ids = {
        _read_json(path)["batch_id"]
        for directory in (paths.pending, paths.dispatched)
        for path in directory.glob("*.json")
    }
    return bool(batch_ids) and all(
        _read_json(paths.batches / f"{batch_id}.json")["sealed"]
        for batch_id in batch_ids
    )


def _run_daemon(paths: ServicePaths, instance_token: str) -> int:
    paths.create()
    lock_file = paths.lock.open("a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 2
    _fail_dispatched(paths, "training queue service was interrupted")
    paths.stop.unlink(missing_ok=True)
    paths.engine_ready.unlink(missing_ok=True)
    stopping = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True
        paths.paused.touch()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    scheduler = Path(__file__).with_name("service_scheduler.sh")
    environment = {
        **os.environ,
        "TRAINING_QUEUE_SERVICE_CHILD": "1",
        "TRAINING_QUEUE_SERVICE_STATE_DIR": str(paths.root),
    }
    environment.setdefault("TRAINING_QUEUE_MONITOR_LIGHT_GPUS", "1")
    process = subprocess.Popen(
        ["bash", str(scheduler), str(paths.root)],
        stdin=subprocess.PIPE,
        env=environment,
    )
    assert process.stdin is not None
    feeding: str | None = None
    training_released = False
    last_dispatch = time.monotonic()
    last_reaper_command = time.monotonic()
    return_code = 0
    try:
        while True:
            _collect_results(paths)
            if process.poll() is not None:
                if not stopping:
                    _fail_dispatched(paths, "persistent GPU scheduler exited")
                    return_code = 1
                break

            if feeding is not None:
                acknowledgement = paths.acknowledgements / feeding
                if acknowledgement.exists():
                    acknowledgement.unlink()
                    feeding = None

            if paths.stop.exists():
                stopping = True
                paths.paused.touch()

            active = _count(paths.dispatched)
            if stopping and active == 0:
                process.stdin.close()
                process.wait()
                break

            now = time.monotonic()
            if (
                active > 0
                and feeding is None
                and now - last_reaper_command >= _REAPER_INTERVAL_SECONDS
            ):
                process.stdin.write(b"_service_reap\n")
                process.stdin.flush()
                last_reaper_command = now

            if (
                not stopping
                and not paths.paused.exists()
                and feeding is None
                and paths.engine_ready.exists()
            ):
                selected = _next_job(paths)
                if selected is not None:
                    _, job = selected
                    job["dispatched_at"] = time.time()
                    _atomic_json(paths.dispatched / f"{job['id']}.json", job)
                    process.stdin.write(_scheduler_command(job, paths).encode())
                    process.stdin.flush()
                    feeding = job["id"]
                    last_dispatch = time.monotonic()

            if (
                not training_released
                and _count(paths.dispatched) > 0
                and _count(paths.pending) == 0
                and (
                    _all_batches_sealed(paths)
                    or time.monotonic() - last_dispatch >= 0.5
                )
            ):
                process.stdin.write(b"_release_training\n")
                process.stdin.flush()
                training_released = True

            _write_status(paths, instance_token)
            time.sleep(0.03)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        _collect_results(paths)
        if _count(paths.dispatched):
            _fail_dispatched(paths, "persistent GPU scheduler stopped")
        status = _status_payload(
            paths,
            pid=os.getpid(),
            instance_token=instance_token,
        )
        status["running"] = False
        _atomic_json(paths.status, status)
        paths.stop.unlink(missing_ok=True)
        paths.engine_ready.unlink(missing_ok=True)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    return return_code


def _start(paths: ServicePaths) -> int:
    paths.create()
    if _service_pid(paths) is not None:
        print("training queue service is already running", file=sys.stderr)
        return 2
    paths.stop.unlink(missing_ok=True)
    paths.paused.unlink(missing_ok=True)
    instance_token = uuid.uuid4().hex
    with paths.daemon_log.open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--state-dir",
                str(paths.root),
                "_serve",
                "--instance-token",
                instance_token,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if _service_pid(paths) == process.pid and paths.engine_ready.exists():
            print(process.pid)
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.02)
    print(f"training queue service failed to start; see {paths.daemon_log}", file=sys.stderr)
    return 1


def _require_running(paths: ServicePaths) -> int | None:
    pid = _service_pid(paths)
    if pid is None:
        print("training queue service is not running", file=sys.stderr)
    return pid


def _new_batch(paths: ServicePaths) -> int:
    if _require_running(paths) is None:
        return 2
    batch_id = uuid.uuid4().hex
    _atomic_json(
        paths.batches / f"{batch_id}.json",
        {
            "id": batch_id,
            "jobs": [],
            "sealed": False,
            "submitted_at": time.time(),
        },
    )
    print(batch_id)
    return 0


def _valid_assignment(assignment: str) -> bool:
    name, separator, _ = assignment.partition("=")
    return bool(separator and re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name))


def _forbidden_persisted_name(assignment: str) -> str | None:
    name = assignment.partition("=")[0]
    upper = name.upper()
    queue_internal = (
        upper.startswith(("TRAINING_QUEUE_", "DCN_GPU_", "DCN_RUN_"))
        or upper
        in {
            "DCN_PREPARED_MARKER",
            "DCN_TRAINING_RELEASE",
            "DCN_RUNNER_DATA_READY",
            "CUDA_VISIBLE_DEVICES",
        }
    )
    if queue_internal:
        return f"refusing to persist queue-internal variable: {name}"
    safe_token_names = {"BEGINNING_TOKEN", "CLS_TOKEN", "PADDING_TOKEN"}
    secret_name = re.search(
        r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|COOKIE)(?:_|$)",
        upper,
    ) or re.search(
        r"(?:^|_)(?:(?:API|PRIVATE)_KEY|(?:ACCESS|AUTH|BEARER|REFRESH|ID|OAUTH)_TOKEN)(?:_|$)",
        upper,
    )
    if secret_name and upper not in safe_token_names:
        return f"refusing to persist secret-like variable: {name}"
    return None


def _stable_job(paths: ServicePaths, run: str) -> dict[str, Any] | None:
    stable_key = hashlib.sha256(run.encode()).hexdigest()
    stable_path = paths.stable / f"{stable_key}.json"
    try:
        indexed = _read_json(stable_path)
    except FileNotFoundError:
        indexed = None
    if indexed is not None:
        job_id = indexed["id"]
        candidates = (
            *(paths.pending.glob(f"*-{job_id}.json")),
            paths.dispatched / f"{job_id}.json",
        )
        for path in candidates:
            try:
                return _read_json(path)
            except FileNotFoundError:
                continue
        stable_path.unlink(missing_ok=True)

    for directory in (paths.pending, paths.dispatched):
        for path in directory.glob("*.json"):
            try:
                job = _read_json(path)
            except FileNotFoundError:
                continue
            if job.get("run") == run:
                _atomic_json(stable_path, job)
                return job
    return None


def _same_stable_payload(
    job: dict[str, Any],
    *,
    script: str,
    run: str,
    data_group: str,
    environment: list[str],
) -> bool:
    return (
        job.get("script") == script
        and job.get("run") == run
        and job.get("data_group") == data_group
        and job.get("environment") == environment
    )


def _enqueue_run(
    paths: ServicePaths,
    *,
    batch_id: str,
    script: str,
    run: str,
    data_group: str,
    environment: Sequence[str],
) -> int:
    if _require_running(paths) is None:
        return 2
    assignments = list(environment)
    if assignments[:1] == ["--"]:
        assignments = assignments[1:]
    invalid = next((value for value in assignments if not _valid_assignment(value)), None)
    if invalid is not None:
        print(f"invalid environment assignment: {invalid!r}", file=sys.stderr)
        return 2
    for assignment in assignments:
        forbidden = _forbidden_persisted_name(assignment)
        if forbidden is not None:
            print(forbidden, file=sys.stderr)
            return 2
    if not re.fullmatch(r"[a-zA-Z0-9_.-]*", data_group):
        print(f"invalid training data group: {data_group!r}", file=sys.stderr)
        return 2
    batch_path = paths.batches / f"{batch_id}.json"
    stable_key = hashlib.sha256(run.encode()).hexdigest()
    stable_lock = (paths.stable / f"{stable_key}.lock").open("a+")
    fcntl.flock(stable_lock, fcntl.LOCK_EX)
    try:
        batch_lock = batch_path.with_suffix(".lock").open("a+")
        fcntl.flock(batch_lock, fcntl.LOCK_EX)
        batch = _read_json(batch_path)
    except FileNotFoundError:
        print(f"unknown batch id: {batch_id}", file=sys.stderr)
        return 2
    try:
        if batch["sealed"]:
            print(f"batch is already sealed: {batch_id}", file=sys.stderr)
            return 2
        existing = _stable_job(paths, run)
        if existing is not None:
            if not _same_stable_payload(
                existing,
                script=script,
                run=run,
                data_group=data_group,
                environment=assignments,
            ):
                print(
                    f"stable run already exists with different payload: {run}",
                    file=sys.stderr,
                )
                return 2
            job_id = existing["id"]
            if job_id not in batch["jobs"]:
                batch["jobs"].append(job_id)
                _atomic_json(batch_path, batch)
            print(job_id)
            return 0
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "batch_id": batch_id,
            "script": script,
            "run": run,
            "data_group": data_group,
            "environment": assignments,
            "submitted_at": time.time(),
        }
        filename = f"{time.time_ns():020d}-{job_id}.json"
        _atomic_json(paths.stable / f"{stable_key}.json", job)
        _atomic_json(paths.pending / filename, job)
        batch["jobs"].append(job_id)
        _atomic_json(batch_path, batch)
    finally:
        fcntl.flock(batch_lock, fcntl.LOCK_UN)
        batch_lock.close()
        fcntl.flock(stable_lock, fcntl.LOCK_UN)
        stable_lock.close()
    print(job_id)
    return 0


def _seal_batch(paths: ServicePaths, batch_id: str) -> int:
    batch_path = paths.batches / f"{batch_id}.json"
    try:
        batch_lock = batch_path.with_suffix(".lock").open("a+")
        fcntl.flock(batch_lock, fcntl.LOCK_EX)
        batch = _read_json(batch_path)
    except FileNotFoundError:
        print(f"unknown batch id: {batch_id}", file=sys.stderr)
        return 2
    try:
        batch["sealed"] = True
        batch["sealed_at"] = time.time()
        _atomic_json(batch_path, batch)
    finally:
        fcntl.flock(batch_lock, fcntl.LOCK_UN)
        batch_lock.close()
    return 0


def _wait_batch(paths: ServicePaths, batch_id: str) -> int:
    batch_path = paths.batches / f"{batch_id}.json"
    while True:
        try:
            batch = _read_json(batch_path)
        except FileNotFoundError:
            print(f"unknown batch id: {batch_id}", file=sys.stderr)
            return 2
        if not batch["sealed"]:
            print(f"batch is not sealed: {batch_id}", file=sys.stderr)
            return 2
        failed = [
            job_id for job_id in batch["jobs"]
            if (paths.failed / f"{job_id}.json").exists()
        ]
        finished = sum(
            (paths.completed / f"{job_id}.json").exists()
            or (paths.failed / f"{job_id}.json").exists()
            for job_id in batch["jobs"]
        )
        if finished == len(batch["jobs"]):
            if failed:
                print(f"{len(failed)} run(s) failed in batch {batch_id}", file=sys.stderr)
                return 1
            return 0
        if _service_pid(paths) is None:
            print("training queue service stopped before the batch finished", file=sys.stderr)
            return 2
        time.sleep(0.05)


def _status(paths: ServicePaths, as_json: bool) -> int:
    try:
        status = _read_json(paths.status)
    except (FileNotFoundError, json.JSONDecodeError):
        status = _status_payload(paths, pid=0, instance_token=None)
    status["running"] = _service_pid(paths) is not None
    if as_json:
        print(json.dumps(status, sort_keys=True))
    else:
        state = "running" if status["running"] else "stopped"
        if status["paused"]:
            state += ", paused"
        print(
            f"{state}; active={status['active']} queued={status['queued']} "
            f"completed={status['completed']} failed={status['failed']}"
        )
    return 0 if status["running"] else 1


def _pause(paths: ServicePaths) -> int:
    if _require_running(paths) is None:
        return 2
    paths.paused.touch()
    return 0


def _drain(paths: ServicePaths) -> int:
    if _pause(paths) != 0:
        return 2
    while True:
        try:
            status = _read_json(paths.status)
        except (FileNotFoundError, json.JSONDecodeError):
            return 2
        if status.get("paused") and status.get("active") == 0:
            return 0
        if _service_pid(paths) is None:
            return 2
        time.sleep(0.05)


def _resume(paths: ServicePaths) -> int:
    if _require_running(paths) is None:
        return 2
    paths.paused.unlink(missing_ok=True)
    return 0


def _stop(paths: ServicePaths) -> int:
    pid = _service_pid(paths)
    if pid is None:
        return 0
    paths.paused.touch()
    paths.stop.touch()
    while _pid_is_running(pid):
        time.sleep(0.05)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2] / "generated" / "training-queue-service"
    parser.add_argument("--state-dir", type=Path, default=default_root)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("start")
    serve = subparsers.add_parser("_serve")
    serve.add_argument("--instance-token", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    subparsers.add_parser("new-batch")
    enqueue = subparsers.add_parser("enqueue-run")
    enqueue.add_argument("--batch", required=True)
    enqueue.add_argument("--script", required=True)
    enqueue.add_argument("--run", required=True)
    enqueue.add_argument("--data-group", default="")
    enqueue.add_argument("environment", nargs=argparse.REMAINDER)
    seal = subparsers.add_parser("seal-batch")
    seal.add_argument("batch_id")
    wait = subparsers.add_parser("wait-batch")
    wait.add_argument("batch_id")
    subparsers.add_parser("pause")
    subparsers.add_parser("drain")
    subparsers.add_parser("resume")
    subparsers.add_parser("stop")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    paths = ServicePaths(arguments.state_dir.resolve())
    paths.create()
    if arguments.action == "start":
        return _start(paths)
    if arguments.action == "_serve":
        return _run_daemon(paths, arguments.instance_token)
    if arguments.action == "status":
        return _status(paths, arguments.json)
    if arguments.action == "new-batch":
        return _new_batch(paths)
    if arguments.action == "enqueue-run":
        return _enqueue_run(
            paths,
            batch_id=arguments.batch,
            script=arguments.script,
            run=arguments.run,
            data_group=arguments.data_group,
            environment=arguments.environment,
        )
    if arguments.action == "seal-batch":
        return _seal_batch(paths, arguments.batch_id)
    if arguments.action == "wait-batch":
        return _wait_batch(paths, arguments.batch_id)
    if arguments.action == "pause":
        return _pause(paths)
    if arguments.action == "drain":
        return _drain(paths)
    if arguments.action == "resume":
        return _resume(paths)
    if arguments.action == "stop":
        return _stop(paths)
    raise AssertionError(arguments.action)


if __name__ == "__main__":
    raise SystemExit(main())
