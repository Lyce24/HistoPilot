"""Bounded lists: `--limit` items from `--offset`, with `hasMore` (docs/cli-contract.md#output)."""

DEFAULT_LIMIT = 50
MAX_LIMIT = 10_000


def served(
    rows: list,
    *,
    offset: int,
    limit: int,
    total: int | None = None,
    has_more: bool | None = None,
) -> dict:
    """The bounds of a page the service cut itself: its own hasMore, else its total."""
    more = has_more if has_more is not None else total is not None and offset + len(rows) < total
    return {"offset": offset, "limit": limit, "hasMore": bool(more)}


def page(items: list, *, offset: int = 0, limit: int = DEFAULT_LIMIT) -> tuple[list, dict]:
    """A page of a list the service returns whole."""
    return (
        items[offset : offset + limit],
        {"offset": offset, "limit": limit, "hasMore": len(items) > offset + limit},
    )
