"""Which commit a checkout runs, read from git's files without spawning git."""

from histopilot.version_info import revision_of

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_a_branch_is_read_from_its_loose_or_packed_ref(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text(f"# pack-refs\n{COMMIT} refs/heads/main\n")
    assert revision_of(tmp_path) == COMMIT
    (git / "refs" / "heads" / "main").write_text("f" * 40 + "\n")
    assert revision_of(tmp_path) == "f" * 40


def test_a_worktree_reads_its_head_and_the_shared_refs(tmp_path):
    common = tmp_path / "main" / ".git"
    worktree = common / "worktrees" / "dev"
    worktree.mkdir(parents=True)
    (worktree / "HEAD").write_text("ref: refs/heads/dev\n")
    (worktree / "commondir").write_text("../..\n")
    (common / "packed-refs").write_text(f"{COMMIT} refs/heads/dev\n")
    checkout = tmp_path / "dev"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {worktree}\n")
    assert revision_of(checkout) == COMMIT


def test_a_detached_head_or_no_checkout(tmp_path):
    assert revision_of(tmp_path) is None
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text(f"{COMMIT}\n")
    assert revision_of(tmp_path) == COMMIT
