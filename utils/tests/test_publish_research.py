import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLISH_SCRIPT = ROOT / "publish_research.sh"
GIT_ENVIRONMENT = os.environ | {
    "GIT_AUTHOR_NAME": "Test Publisher",
    "GIT_AUTHOR_EMAIL": "publisher@example.com",
    "GIT_COMMITTER_NAME": "Test Publisher",
    "GIT_COMMITTER_EMAIL": "publisher@example.com",
}


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    subprocess.run(
        ["git", "init", "-b", "main", source],
        check=True,
        capture_output=True,
        text=True,
    )
    shutil.copy2(PUBLISH_SCRIPT, source / PUBLISH_SCRIPT.name)
    (source / ".gitignore").write_text("cache/\n")
    (source / ".publishignore").write_text(
        "/week02_version.ipynb\nprivate.txt\nprivate/\nbare-private\n"
    )
    (source / "result.md").write_text("public result\n")
    (source / "cache").mkdir()
    (source / "cache" / "local.bin").write_text("cache\n")
    (source / "cache" / "published.bin").write_text("tracked cache\n")
    (source / "week02_version.ipynb").write_text("homework\n")
    (source / "private.txt").write_text("private\n")
    subprocess.run(
        ["git", "-C", source, "add", "-f", "private.txt", "cache/published.bin"],
        check=True,
    )
    (source / "nested" / "private").mkdir(parents=True)
    (source / "nested" / "private" / "value.txt").write_text("private\n")
    (source / "nested" / "bare-private").mkdir()
    (source / "nested" / "bare-private" / "value.txt").write_text("private\n")
    (source / "directory").mkdir()
    (source / "directory" / "child.txt").write_text("child\n")
    (source / "single").write_text("file\n")
    (source / "mode.sh").write_text("#!/usr/bin/env bash\n")
    (source / "mode.sh").chmod(0o755)
    (source / "local-secret.txt").write_text("private\n")
    with (source / ".git" / "info" / "exclude").open("a") as git_exclude:
        git_exclude.write("local-secret.txt\n")
    return source


def _destination(tmp_path: Path) -> Path:
    destination = tmp_path / "destination"
    subprocess.run(
        ["git", "init", "-b", "main", destination],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination


def _run(
    script: Path, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [script, *arguments],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _commit(destination: Path, *paths: str) -> None:
    subprocess.run(["git", "-C", destination, "add", *paths], check=True)
    subprocess.run(
        ["git", "-C", destination, "commit", "-m", "Destination state"],
        check=True,
        capture_output=True,
        text=True,
        env=GIT_ENVIRONMENT,
    )


def test_publish_dry_run_previews_and_apply_mirrors_public_files(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = _destination(tmp_path)
    (destination / ".gitignore").write_text("cache/\n")
    (destination / "stale.txt").write_text("stale\n")
    (destination / "cache").mkdir()
    (destination / "cache" / "old.bin").write_text("old cache\n")
    (destination / "week02_version.ipynb").write_text("old homework\n")
    (destination / "directory").write_text("old file\n")
    (destination / "single").mkdir()
    (destination / "single" / "child.txt").write_text("old child\n")
    (destination / "mode.sh").write_text("#!/usr/bin/env bash\n")
    _commit(
        destination,
        ".gitignore",
        "stale.txt",
        "week02_version.ipynb",
        "directory",
        "single/child.txt",
        "mode.sh",
    )
    script = source / PUBLISH_SCRIPT.name

    preview = _run(script, "--destination", str(destination))

    assert not (destination / "result.md").exists()
    assert (destination / "stale.txt").exists()
    assert "> mode.sh" in preview.stdout

    _run(script, "--apply", "--destination", str(destination))

    assert (destination / "result.md").read_text() == "public result\n"
    assert not (destination / "stale.txt").exists()
    assert (destination / "cache" / "old.bin").exists()
    assert not (destination / "week02_version.ipynb").exists()
    assert not (destination / "private.txt").exists()
    assert not (destination / "nested" / "private").exists()
    assert not (destination / "nested" / "bare-private").exists()
    assert not (destination / "local-secret.txt").exists()
    assert not (destination / "cache" / "local.bin").exists()
    assert (destination / "cache" / "published.bin").read_text() == "tracked cache\n"
    assert (destination / "directory" / "child.txt").read_text() == "child\n"
    assert (destination / "single").read_text() == "file\n"
    assert (destination / "mode.sh").stat().st_mode & 0o111
    assert (destination / ".git").is_dir()


def test_publish_pushes_a_commit(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = _destination(tmp_path)
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", remote],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", destination, "remote", "add", "origin", str(remote)],
        check=True,
    )
    _run(
        source / PUBLISH_SCRIPT.name,
        "--push",
        "--destination",
        str(destination),
        "--message",
        "Publish test results",
        env=GIT_ENVIRONMENT,
    )

    commit = subprocess.run(
        ["git", "--git-dir", remote, "rev-parse", "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subject = subprocess.run(
        ["git", "--git-dir", remote, "show", "-s", "--format=%s", commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert subject == "Publish test results"
    published_cache = subprocess.run(
        ["git", "--git-dir", remote, "show", f"{commit}:cache/published.bin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert published_cache == "tracked cache\n"


def test_publish_refuses_destination_changes_and_path_overlap(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = _destination(tmp_path)
    (destination / "notes.txt").write_text("local work\n")

    changed = subprocess.run(
        [source / PUBLISH_SCRIPT.name, "--apply", "--destination", destination],
        text=True,
        capture_output=True,
    )

    assert changed.returncode != 0
    assert (destination / "notes.txt").read_text() == "local work\n"

    ancestor = tmp_path / "ancestor"
    subprocess.run(
        ["git", "init", "-b", "main", ancestor],
        check=True,
        capture_output=True,
        text=True,
    )
    nested_source = ancestor / "source"
    shutil.copytree(source, nested_source, symlinks=True)

    overlapping = subprocess.run(
        [nested_source / PUBLISH_SCRIPT.name, "--apply", "--destination", ancestor],
        text=True,
        capture_output=True,
    )

    assert overlapping.returncode != 0
    assert nested_source.is_dir()
