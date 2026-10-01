"""What an AI agent may see of a project shared at the `metadata` exposure level.

JSON only: patient and slide identifiers become stable keyed pseudonyms, and file paths,
free text a person wrote and per-case values are withheld. An identifier is found by the
key that holds it (any key naming slide, patient or case IDs), by a list made mostly of the
project's known identifiers, and by sweeping every remaining string for known identifiers,
which also catches names such as "Attention · <slide ID>" and file names such as
"<slide ID>.tiff". Short all-digit identifiers are not swept from text, where they read as
counts; keys and per-case lists still replace them. A map keyed by cases, or a list of
records each about one case, is per-case: only its size is shown. Hashes and record IDs
stay intact, so a person confirms the very preview an agent saw. This lowers risk; it is
never de-identification.
"""

import base64
import hashlib
import hmac
import re

# Keys whose values are dropped outright. Keys are compared without case, "_" or "-", so
# snake_case keys from other tools (TRIDENT's `wsi_rel_paths`, `job_dir`) match too.
PATH_KEYS = frozenset({"path", "paths", "cwd", "argv", "env", "log", "logtail", "sourcepath"})
PATH_SUFFIXES = (
    "path",
    "paths",
    "folder",
    "folders",
    "root",
    "roots",
    "directory",
    "directories",
    "dir",
    "dirs",
)
# Free text a person wrote, by key; the service's own notes and messages stay.
TEXT_KEYS = frozenset(
    {
        "notes",
        "reviewer",
        "reviewers",
        "reasons",
        "description",
        "example",
        "comment",
        "comments",
        "remark",
        "remarks",
    }
)
# Free text inside a small record, by that record's key: a version's note.
TEXT_INSIDE = {"versionlabel": frozenset({"note"})}
CASE_KEYS = frozenset({"attributes"})
# Keys whose values are case identifiers, by the end of their name, with a pseudonym prefix:
# slideId, slideIds, selectedSlideIds, missingPackSlideIds, patient_id, caseId, currentSlide.
ID_SUFFIXES = (
    ("slideids", "S"),
    ("slideid", "S"),
    ("patientids", "P"),
    ("patientid", "P"),
    ("caseids", "P"),
    ("caseid", "P"),
    ("currentslide", "S"),
)
# Text values of these keys are kept byte for byte: confirmations compare them.
KEPT_SUFFIXES = ("Hash", "hash", "sha256", "Sha256")
# All-digit identifiers shorter than this are not swept from text.
MIN_SWEPT_ID = 4
# A list or map with at least this many entries, this share of them about known cases, is
# per-case, whatever its key: a list's identifiers are replaced even when short, and a map's
# or a list of records' values are withheld.
PER_CASE_MIN = 5
PER_CASE_SHARE = 0.8
WITHHELD = "per-case values"
_PATH = re.compile(r"(?<![\w.~-])(?:/[^\s/\"'<>`|,;]+){2,}/?")


def _normal(key: str) -> str:
    return key.replace("_", "").replace("-", "").lower()


def _dropped(key: str) -> bool:
    name = _normal(key)
    return (
        name in PATH_KEYS
        or name in TEXT_KEYS
        or name in CASE_KEYS
        or (name.endswith(PATH_SUFFIXES) and _id_prefix(key) is None)
    )


def _id_prefix(key: str) -> str | None:
    name = _normal(key)
    return next((prefix for suffix, prefix in ID_SUFFIXES if name.endswith(suffix)), None)


def _sweepable(identifier: str) -> bool:
    """Long identifiers, and short ones mixing letters and digits such as T1 or S07."""
    if len(identifier) >= MIN_SWEPT_ID:
        return True
    return (
        len(identifier) >= 2
        and any(character.isalpha() for character in identifier)
        and any(character.isdigit() for character in identifier)
    )


class Redactor:
    def __init__(self, key: bytes, known: dict[str, str] | None = None):
        """``known`` maps a project's case identifiers to their prefix, P or S."""
        self.key = key
        self.known = {
            value: prefix
            for value, prefix in (known or {}).items()
            if isinstance(value, str) and value
        }
        names = sorted((name for name in self.known if _sweepable(name)), key=len, reverse=True)
        self._sweep = (
            re.compile(
                "(?<![A-Za-z0-9])("
                + "|".join(re.escape(name) for name in names)
                + ")(?![A-Za-z0-9])"
            )
            if names
            else None
        )

    def pseudonym(self, value: str, prefix: str) -> str:
        # A known identifier keeps one pseudonym wherever it appears, whatever its key.
        prefix = self.known.get(value, prefix)
        digest = hmac.new(self.key, f"{prefix}:{value}".encode(), hashlib.sha256).digest()
        return f"{prefix}-{base64.b32encode(digest).decode()[:10]}"

    def text(self, value: str) -> str:
        # Identifiers before paths: a path ends at whitespace, so replacing it first would
        # leave the tail of a file name such as "Case 12 B1.h5" behind the <path>.
        if self._sweep is not None:
            value = self._sweep.sub(
                lambda match: self.pseudonym(match[1], self.known[match[1]]), value
            )
        return _PATH.sub("<path>", value)

    def document(self, value):
        if isinstance(value, dict):
            if self._per_case([self._case_named(key) for key in value]):
                return {"withheld": WITHHELD, "count": len(value)}
            return {
                self.text(key) if isinstance(key, str) else key: self._field(key, item)
                for key, item in value.items()
                if not _dropped(str(key))
            }
        if isinstance(value, list):
            if self._per_case([self._case_of(item) for item in value]):
                if any(isinstance(item, dict) for item in value):
                    return {"withheld": WITHHELD, "count": len(value)}
                return [
                    self.pseudonym(item, "S") if self._is_known(item) else self.document(item)
                    for item in value
                ]
            return [self.document(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value

    def _is_known(self, value) -> bool:
        return isinstance(value, str) and value in self.known

    def _case_named(self, text):
        """The known case a key or list item names: itself, or the case a path or file name
        in it names, such as the slide of `/features/<slide ID>.h5`."""
        if not isinstance(text, str) or text in self.known:
            return text
        match = self._sweep.search(text) if self._sweep is not None else None
        if match:
            return match[1]
        stem = text.rsplit("/", 1)[-1].split(".", 1)[0]
        return stem if stem in self.known else None

    def _case_of(self, item):
        """The case a list item is about: the case it names, or the known case a record
        names."""
        if not isinstance(item, dict):
            return self._case_named(item)
        return next(
            (
                value
                for key, value in item.items()
                if (key == "id" or _id_prefix(str(key))) and self._is_known(value)
            ),
            None,
        )

    def _per_case(self, values: list) -> bool:
        if len(values) < PER_CASE_MIN:
            return False
        return sum(self._is_known(item) for item in values) >= PER_CASE_SHARE * len(values)

    def _field(self, key, value):
        key = str(key)
        prefix = _id_prefix(key)
        if prefix is not None:
            return self._identifier(value, prefix)
        if key.endswith(KEPT_SUFFIXES) and not isinstance(value, dict | list):
            return value
        # A record's own ID stays; a case's ID, such as a case review row's, does not.
        if key == "id" and self._is_known(value):
            return self.pseudonym(value, "P")
        inside = TEXT_INSIDE.get(_normal(key))
        if inside and isinstance(value, dict):
            value = {name: item for name, item in value.items() if _normal(str(name)) not in inside}
        return self.document(value)

    def _identifier(self, value, prefix: str):
        if isinstance(value, str) and value:
            return self.pseudonym(value, prefix)
        if isinstance(value, list):
            return [self._identifier(item, prefix) for item in value]
        return self.document(value)
