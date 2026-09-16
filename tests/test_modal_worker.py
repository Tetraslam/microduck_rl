from pathlib import Path

import pytest

from mjlab_microduck.modal_worker import atomic_checkpoint, resolve_checkpoint


def test_failed_save_keeps_previous_checkpoint(tmp_path):
    target = tmp_path / "model_50.pt"
    target.write_bytes(b"previous complete checkpoint")
    commits = []

    def interrupted_write(path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted writer")

    with pytest.raises(RuntimeError, match="interrupted writer"):
        atomic_checkpoint(str(target), interrupted_write, lambda: commits.append(True))
    assert target.read_bytes() == b"previous complete checkpoint"
    assert not commits
    assert not list(tmp_path.glob("*.incomplete"))


def test_commit_only_sees_complete_checkpoint(tmp_path):
    target = tmp_path / "model_50.pt"
    snapshots = []
    atomic_checkpoint(
        str(target),
        lambda p: Path(p).write_bytes(b"complete"),
        lambda: snapshots.append(target.read_bytes()),
    )
    assert snapshots == [b"complete"]


def test_commit_failure_is_not_reported_as_success(tmp_path):
    target = tmp_path / "model_50.pt"

    def failed_commit():
        raise RuntimeError("volume unavailable")

    with pytest.raises(RuntimeError, match="volume unavailable"):
        atomic_checkpoint(
            str(target), lambda p: Path(p).write_bytes(b"complete"), failed_commit
        )
    assert target.read_bytes() == b"complete"


def test_resume_rejects_paths_outside_volume_and_partial_files(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        resolve_checkpoint(tmp_path, "../model_50.pt")
    with pytest.raises(ValueError, match="named"):
        resolve_checkpoint(tmp_path, "model_50.pt.incomplete")
