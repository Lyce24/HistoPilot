"""One server-side population drives both table pagination and descriptive charts."""

from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext

from pydantic import Field

from histopilot.schemas.workspace import RequestModel
from histopilot.storage.project_lock import StorageError


class AttributeFilter(RequestModel):
    field: str = Field(min_length=1, max_length=128)
    values: list[str | None] = Field(min_length=1, max_length=100)


class ExploreRequest(RequestModel):
    field: str | None = Field(default=None, max_length=128)
    compare: str | None = Field(default=None, max_length=128)
    search: str = Field(default="", max_length=256)
    filters: list[AttributeFilter] = Field(default_factory=list, max_length=30)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=1000)


def explore(records: list[dict], dictionary: list[dict], request: ExploreRequest) -> dict:
    fields = {item["key"]: item for item in dictionary}
    for key in [request.field, request.compare, *(item.field for item in request.filters)]:
        if key and key not in fields:
            raise StorageError(f"Unknown attribute: {key}.", "UNKNOWN_ATTRIBUTE", 422)

    def value(row, key):
        raw = row["attributes"].get(key)
        return None if raw is None else str(raw)

    search = request.search.casefold()
    selected = [
        row
        for row in records
        if all(value(row, item.field) in item.values for item in request.filters)
        and (
            not search
            or search
            in " ".join(
                [
                    row["slideId"],
                    row.get("patientId") or "",
                    *(str(item) for item in row["attributes"].values() if item is not None),
                ]
            ).casefold()
        )
    ]

    def population(keys):
        if keys and all(fields[key]["owner"] == "patient" for key in keys):
            unique = {row["patientId"]: row for row in selected if row.get("patientId")}
            unit = (
                "group"
                if any(row.get("patientIdSource") == "slide_fallback" for row in unique.values())
                else "patient"
            )
            return list(unique.values()), unit
        return selected, "slide"

    unlinked = sum(not row.get("patientId") for row in selected)
    raw_values = (
        Counter(value(row, request.field) for row in selected) if request.field else Counter()
    )
    ordered_values = sorted(
        raw_values.items(), key=lambda pair: (-pair[1], pair[0] is not None, pair[0] or "")
    )
    distribution = {
        "kind": "none",
        "unit": "slide",
        "counts": [],
        "missingCount": 0,
        "total": 0,
        "unlinkedSlideCount": 0,
        "truncated": False,
        "otherCount": 0,
        "omittedCategoryCount": 0,
    }
    if request.field:
        attribute = fields[request.field]
        rows, unit = population([request.field])
        raw = [value(row, request.field) for row in rows]
        distribution.update(
            unit=unit,
            total=len(raw),
            missingCount=sum(item is None for item in raw),
            unlinkedSlideCount=unlinked if unit in {"patient", "group"} else 0,
        )
        if attribute["type"] in {"categorical", "ordered_categorical", "boolean"}:
            counts = Counter(raw)
            if attribute["type"] == "ordered_categorical":
                order = {key: index for index, key in enumerate(attribute.get("categories") or [])}
                items = sorted(
                    counts.items(), key=lambda pair: (order.get(pair[0], len(order)), pair[0] or "")
                )
            else:
                items = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0] or ""))
            distribution.update(
                kind="categorical",
                counts=[{"value": key, "count": count} for key, count in items[:100]],
                truncated=len(items) > 100,
                otherCount=sum(count for _, count in items[100:]),
                omittedCategoryCount=max(0, len(items) - 100),
            )
        elif attribute["type"] in {"integer", "decimal"}:
            bins = []
            try:
                # Decimal avoids overflow in high-low, underflow in bin width and
                # merging adjacent integer values beyond binary64 precision.
                numbers = [Decimal(item) for item in raw if item is not None]
                if any(not item.is_finite() for item in numbers):
                    raise InvalidOperation
                if numbers:
                    with localcontext() as context:
                        context.prec = max(
                            34, *(len(item.as_tuple().digits) + 4 for item in numbers)
                        )
                        low, high = min(numbers), max(numbers)
                        size = min(10, len(set(numbers)))
                        width = (high - low) / size if high != low else Decimal(1)
                        counts = Counter(
                            min(size - 1, int((number - low) / width)) for number in numbers
                        )
                        bins = [
                            {
                                "value": f"{low + index * width:g} – {min(high, low + (index + 1) * width):g}",
                                "count": counts[index],
                            }
                            for index in range(size)
                        ]
            except (InvalidOperation, ValueError, OverflowError, ZeroDivisionError) as error:
                raise StorageError(
                    "A numeric attribute contains invalid or unsupported values.",
                    "INVALID_ATTRIBUTE",
                    422,
                ) from error
            distribution.update(kind="numeric", counts=bins)
    cross_tab = None
    categorical = {"categorical", "ordered_categorical", "boolean"}
    if (
        request.field
        and request.compare
        and all(fields[key]["type"] in categorical for key in [request.field, request.compare])
    ):
        rows, unit = population([request.field, request.compare])
        # Missing stays JSON null, distinct from a literal source value '(missing)'.
        pairs = Counter((value(row, request.field), value(row, request.compare)) for row in rows)
        first = sorted({pair[0] for pair in pairs}, key=lambda key: key or "")[:20]
        second = sorted({pair[1] for pair in pairs}, key=lambda key: key or "")[:20]
        cross_tab = {
            "rows": first,
            "columns": second,
            "counts": [[pairs[(left, right)] for right in second] for left in first],
            "unit": unit,
            "total": len(rows),
            "displayedCount": sum(pairs[(left, right)] for left in first for right in second),
            "omittedCount": sum(
                count
                for (left, right), count in pairs.items()
                if left not in first or right not in second
            ),
            "truncated": len({pair[0] for pair in pairs}) > 20
            or len({pair[1] for pair in pairs}) > 20,
            "unlinkedSlideCount": unlinked if unit in {"patient", "group"} else 0,
        }
    return {
        "records": selected[request.offset : request.offset + request.limit],
        "total": len(selected),
        "offset": request.offset,
        "limit": request.limit,
        "totalSlides": len(records),
        "distribution": distribution,
        "crossTab": cross_tab,
        "valueCounts": [{"value": item, "count": count} for item, count in ordered_values[:200]],
        "valuesTruncated": len(ordered_values) > 200,
        "valueCountsUnit": "slide",
        "summary": {
            "slideCount": len(selected),
            "mappedPatientCount": len(
                {row["patientId"] for row in selected if row.get("patientId")}
            ),
            "unlinkedSlideCount": unlinked,
            "verifiedPatientCount": len(
                {
                    row["patientId"]
                    for row in selected
                    if row.get("patientId") and row.get("patientIdSource") != "slide_fallback"
                }
            ),
            "fallbackSlideCount": sum(
                row.get("patientIdSource") == "slide_fallback" for row in selected
            ),
            "groupCount": len({row["patientId"] for row in selected if row.get("patientId")}),
        },
    }
