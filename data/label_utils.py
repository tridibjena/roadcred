"""IDD label handling, keyed by class *name* rather than by numeric ID.

IDD ships several parallel label encodings (raw ``id``, ``level4Id``, ``level3Id``,
``level2Id``, ``level1Id``) and different releases render different ones to disk --
IDD Lite ships ``*_gtFine_labellevel1Ids.png`` while the full IDD Segmentation release
ships ``*_gtFine_polygons.json``. The numeric ID for a given class is therefore *not*
stable across releases or across levels.

Everything here is consequently driven by the frozen set of drivable class **names**
(:data:`DRIVABLE_NAMES`); numeric lookup tables are derived from the official label table
at import time. If a future release renames or adds a class, :func:`load_label_table`
picks up the dataset's own ``anue_labels.py`` / ``labels.csv`` when one is present in the
dataset root, and only falls back to the embedded copy otherwise.

Two target class spaces are supported, both derived from the same table:

``level1`` (the project's main task, 7 classes -- IDD's own level-1 hierarchy)::

    0 drivable   1 non-drivable   2 living-thing   3 vehicles
    4 barrier/structures   5 construction/vegetation   6 sky
    255 ignore (IDD void classes, excluded from the loss)

``binary`` (a derived 2-class baseline: is this pixel drivable?)::

    0 nondrivable   1 drivable   255 ignore
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np

#: Binary class indices.
NONDRIVABLE, DRIVABLE = 0, 1
IGNORE_INDEX = 255

#: The 7 level-1 class names, indexed by ``level1Id``. These are IDD's own top-level
#: hierarchy, so the main task uses IDD's label semantics unmodified.
LEVEL1_NAMES: tuple[str, ...] = (
    "drivable",
    "non-drivable",
    "living-thing",
    "vehicles",
    "barrier-structures",
    "construction-vegetation",
    "sky",
)


class IDDLabel(NamedTuple):
    """One row of the IDD label definition table."""

    name: str
    id: int
    level4Id: int
    level3Id: int
    category: str
    level2Id: int
    level1Id: int
    ignoreInEval: bool


# Official IDD label table, transcribed from AutoNUE/public-code helpers/anue_labels.py
# (https://github.com/AutoNUE/public-code). Used only as a fallback: if the downloaded
# dataset ships its own label definition, load_label_table() prefers that.
_EMBEDDED_TABLE: tuple[IDDLabel, ...] = (
    IDDLabel("road",                  0, 0,  0,  "drivable",       0,  0,  False),
    IDDLabel("parking",               1, 1,  1,  "drivable",       1,  0,  False),
    IDDLabel("drivable fallback",     2, 2,  1,  "drivable",       1,  0,  False),
    IDDLabel("sidewalk",              3, 3,  2,  "non-drivable",   2,  1,  False),
    IDDLabel("rail track",            4, 3,  3,  "non-drivable",   3,  1,  False),
    IDDLabel("non-drivable fallback", 5, 4,  3,  "non-drivable",   3,  1,  False),
    IDDLabel("person",                6, 5,  4,  "living-thing",   4,  2,  False),
    IDDLabel("animal",                7, 6,  4,  "living-thing",   4,  2,  True),
    IDDLabel("rider",                 8, 7,  5,  "living-thing",   5,  2,  False),
    IDDLabel("motorcycle",            9, 8,  6,  "2-wheeler",      6,  3,  False),
    IDDLabel("bicycle",              10, 9,  7,  "2-wheeler",      6,  3,  False),
    IDDLabel("autorickshaw",         11, 10, 8,  "autorickshaw",   7,  3,  False),
    IDDLabel("car",                  12, 11, 9,  "car",            7,  3,  False),
    IDDLabel("truck",                13, 12, 10, "large-vehicle",  8,  3,  False),
    IDDLabel("bus",                  14, 13, 11, "large-vehicle",  8,  3,  False),
    IDDLabel("caravan",              15, 14, 12, "large-vehicle",  8,  3,  True),
    IDDLabel("trailer",              16, 15, 12, "large-vehicle",  8,  3,  True),
    IDDLabel("train",                17, 15, 12, "large-vehicle",  8,  3,  True),
    IDDLabel("vehicle fallback",     18, 15, 12, "large-vehicle",  8,  3,  False),
    IDDLabel("curb",                 19, 16, 13, "barrier",        9,  4,  False),
    IDDLabel("wall",                 20, 17, 14, "barrier",        9,  4,  False),
    IDDLabel("fence",                21, 18, 15, "barrier",        10, 4,  False),
    IDDLabel("guard rail",           22, 19, 16, "barrier",        10, 4,  False),
    IDDLabel("billboard",            23, 20, 17, "structures",     11, 4,  False),
    IDDLabel("traffic sign",         24, 21, 18, "structures",     11, 4,  False),
    IDDLabel("traffic light",        25, 22, 19, "structures",     11, 4,  False),
    IDDLabel("pole",                 26, 23, 20, "structures",     12, 4,  False),
    IDDLabel("polegroup",            27, 23, 20, "structures",     12, 4,  False),
    IDDLabel("obs-str-bar-fallback", 28, 24, 21, "structures",     12, 4,  False),
    IDDLabel("building",             29, 25, 22, "construction",   13, 5,  False),
    IDDLabel("bridge",               30, 26, 23, "construction",   13, 5,  False),
    IDDLabel("tunnel",               31, 26, 23, "construction",   13, 5,  False),
    IDDLabel("vegetation",           32, 27, 24, "vegetation",     14, 5,  False),
    IDDLabel("sky",                  33, 28, 25, "sky",            15, 6,  False),
    IDDLabel("fallback background",  34, 29, 25, "object fallback",15, 6,  False),
    IDDLabel("unlabeled",            35, 255, 255, "void",         255, 255, True),
    IDDLabel("ego vehicle",          36, 255, 255, "void",         255, 255, True),
    IDDLabel("rectification border", 37, 255, 255, "void",         255, 255, True),
    IDDLabel("out of roi",           38, 255, 255, "void",         255, 255, True),
    IDDLabel("license plate",        39, 255, 255, "vehicle",      255, 255, True),
)

#: The frozen definition of "drivable" for this project, as class names.
#: These are exactly IDD's ``drivable`` category, i.e. every label with ``level1Id == 0``.
DRIVABLE_NAMES: frozenset[str] = frozenset({"road", "parking", "drivable fallback"})

#: Which label-ID column a rendered mask uses, inferred from its filename.
LEVEL_FIELDS = {
    "id": "id",
    "level4Ids": "level4Id",
    "level3Ids": "level3Id",
    "level2Ids": "level2Id",
    "level1Ids": "level1Id",
}


def _parse_anue_labels(source: str) -> tuple[IDDLabel, ...]:
    """Parse an ``anue_labels.py``-style table into :class:`IDDLabel` rows."""
    rows: list[IDDLabel] = []
    pattern = re.compile(r"Label\(\s*'([^']+)'\s*,(.+?)\)\s*,?\s*$", re.MULTILINE)
    for match in pattern.finditer(source):
        name = match.group(1)
        # Fields after the name: id, csId, csTrainId, level4id, level3Id,
        # category, level2Id, level1Id, hasInstances, ignoreInEval, color
        rest = match.group(2)
        rest = re.sub(r"\([^)]*\)", "COLOR", rest)  # collapse the RGB tuple
        parts = [p.strip().strip("'") for p in rest.split(",")]
        if len(parts) < 10:
            continue
        rows.append(
            IDDLabel(
                name=name,
                id=int(parts[0]),
                level4Id=int(parts[3]),
                level3Id=int(parts[4]),
                category=parts[5],
                level2Id=int(parts[6]),
                level1Id=int(parts[7]),
                ignoreInEval=parts[9].lower().startswith("t"),
            )
        )
    return tuple(rows)


def load_label_table(dataset_root: str | Path | None = None) -> tuple[IDDLabel, ...]:
    """Return the IDD label table, preferring one shipped with the dataset.

    Args:
        dataset_root: Root of a downloaded IDD tree. Searched (shallowly) for an
            ``anue_labels.py``; if found, that definition wins over the embedded copy
            so that a future IDD release with renamed or added classes is honoured.

    Returns:
        The label table as a tuple of :class:`IDDLabel`.
    """
    if dataset_root is not None:
        root = Path(dataset_root)
        for candidate in list(root.glob("anue_labels.py")) + list(
            root.glob("*/anue_labels.py")
        ):
            parsed = _parse_anue_labels(candidate.read_text())
            if parsed:
                return parsed
    return _EMBEDDED_TABLE


def drivable_names(table: Iterable[IDDLabel] | None = None) -> frozenset[str]:
    """Names IDD considers drivable, derived from the table's ``drivable`` category.

    Cross-checks against the frozen :data:`DRIVABLE_NAMES`; if a release introduces a new
    drivable class the union is returned, so new classes are picked up rather than
    silently dropped into ``nondrivable``.
    """
    table = table or _EMBEDDED_TABLE
    derived = {label.name for label in table if label.category == "drivable"}
    return frozenset(derived | DRIVABLE_NAMES)


def build_lut(
    level: str = "level3Ids",
    target: str = "level1",
    table: Iterable[IDDLabel] | None = None,
) -> np.ndarray:
    """Build a 256-entry lookup table from raw IDD IDs to the target class space.

    Args:
        level: Which IDD ID column the source mask encodes; one of
            ``id`` / ``level4Ids`` / ``level3Ids`` / ``level2Ids`` / ``level1Ids``.
        target: ``level1`` for the 7-class task, or ``binary`` for drivable/non-drivable.
        table: Label table; defaults to the embedded official one.

    Returns:
        ``uint8`` array of length 256 mapping ``lut[raw_id]`` into the target space.
        IDs IDD never assigns map to :data:`IGNORE_INDEX` rather than to a real class, so
        an unexpected value can never be silently scored as a correct prediction.

    Note:
        For ``level="level1Ids", target="level1"`` this is the identity on 0-6 -- which is
        exactly the IDD Lite case -- but it is still built from the table rather than
        assumed, so the same code path serves full-IDD level3 masks unchanged.
    """
    if level not in LEVEL_FIELDS:
        raise ValueError(f"Unknown level {level!r}; expected one of {sorted(LEVEL_FIELDS)}")
    if target not in {"level1", "binary"}:
        raise ValueError(f"Unknown target {target!r}; expected 'level1' or 'binary'")
    field = LEVEL_FIELDS[level]
    table = tuple(table or _EMBEDDED_TABLE)
    names = drivable_names(table)

    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for label in table:
        raw_id = getattr(label, field)
        if raw_id == IGNORE_INDEX or label.category == "void":
            continue  # stays ignore
        # A level ID can be shared by several names (e.g. level1Id 0 covers every
        # drivable class). Only assign a class when all names sharing the ID agree.
        sharing = [other for other in table if getattr(other, field) == raw_id]
        if target == "binary":
            lut[raw_id] = (
                DRIVABLE if all(other.name in names for other in sharing) else NONDRIVABLE
            )
        else:
            level1_ids = {other.level1Id for other in sharing}
            if len(level1_ids) == 1:
                value = next(iter(level1_ids))
                lut[raw_id] = value if value != IGNORE_INDEX else IGNORE_INDEX
    return lut


def max_id_for_level(level: str, table: Iterable[IDDLabel] | None = None) -> int:
    """Largest non-void ID that ``level`` can legitimately contain.

    Used by :func:`remap_mask` to range-check a mask against its assumed level, so a
    mis-detected level fails loudly instead of silently mis-mapping classes.
    """
    if level not in LEVEL_FIELDS:
        raise ValueError(f"Unknown level {level!r}; expected one of {sorted(LEVEL_FIELDS)}")
    field = LEVEL_FIELDS[level]
    table = table or _EMBEDDED_TABLE
    return max(
        getattr(label, field) for label in table if getattr(label, field) != IGNORE_INDEX
    )


def detect_level(mask_path: str | Path) -> str:
    """Infer the ID encoding of a rendered IDD mask from its path.

    Releases disagree on naming. Some render ``*_gtFine_labellevel3Ids.png``, which names
    the level outright. **IDD Lite does not**: it ships bare ``<frame>_label.png`` files
    that nonetheless contain *level1* IDs (its 7-class "lite" label space). Defaulting
    those to level3 silently maps level1Id 1 -- sidewalk, rail track, non-drivable
    fallback -- onto *drivable*, corrupting the exact boundary this project measures.

    Resolution order: explicit filename marker, then the IDD Lite path marker, then the
    level3 default. :func:`remap_mask` additionally range-checks the result.
    """
    path = Path(mask_path)
    stem = path.stem.lower()
    for level in ("level1Ids", "level2Ids", "level3Ids", "level4Ids"):
        if f"label{level.lower()}" in stem:
            return level
    if stem.endswith("labelids") or "_labelids" in stem:
        return "id"
    if "lite" in str(path).lower():
        return "level1Ids"
    return "level3Ids"


def remap_mask(
    mask: np.ndarray,
    level: str | None = None,
    lut: np.ndarray | None = None,
    target: str = "level1",
) -> np.ndarray:
    """Map a raw IDD ID mask into the target class space.

    Args:
        mask: 2-D array of raw IDD IDs.
        level: ID encoding of ``mask``; ignored when ``lut`` is supplied.
        target: ``level1`` (7 classes) or ``binary``; ignored when ``lut`` is supplied.
        lut: Precomputed lookup table from :func:`build_lut`, to avoid rebuilding it
            per image inside a dataset-wide loop.

    Returns:
        ``uint8`` array of the same shape in the RoadCred class space.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if lut is None:
        level = level or "level3Ids"
        observed = mask[mask != IGNORE_INDEX]
        if observed.size and int(observed.max()) > max_id_for_level(level):
            raise ValueError(
                f"Mask contains ID {int(observed.max())}, which exceeds the maximum "
                f"{max_id_for_level(level)} for level {level!r}. The level was likely "
                "mis-detected; pass an explicit level or check detect_level()."
            )
        lut = build_lut(level, target)
    return lut[mask]


def polygons_to_mask(
    polygons_json: dict,
    table: Iterable[IDDLabel] | None = None,
) -> np.ndarray:
    """Rasterise IDD's ``*_gtFine_polygons.json`` directly into level-1 classes.

    The full IDD Segmentation release ships polygons rather than rendered ID masks.
    Rasterising by name skips the ID indirection entirely, which is the most robust
    path available.

    Args:
        polygons_json: Parsed contents of a ``*_gtFine_polygons.json`` file.
        table: Label table; defaults to the embedded official one.

    Returns:
        ``uint8`` mask of shape ``(imgHeight, imgWidth)`` of level-1 class indices.
    """
    import cv2

    by_name = {label.name: label for label in (table or _EMBEDDED_TABLE)}
    height = int(polygons_json["imgHeight"])
    width = int(polygons_json["imgWidth"])
    # Start as ignore; objects paint over it in file order (IDD lists back-to-front).
    mask = np.full((height, width), IGNORE_INDEX, dtype=np.uint8)

    for obj in polygons_json.get("objects", []):
        if obj.get("deleted"):
            continue
        label = obj.get("label", "")
        points = np.asarray(obj.get("polygon", []), dtype=np.int32)
        if points.shape[0] < 3:
            continue
        entry = by_name.get(label)
        value = IGNORE_INDEX if entry is None else entry.level1Id
        cv2.fillPoly(mask, [points], int(value))
    return mask


__all__ = [
    "IDDLabel",
    "DRIVABLE_NAMES",
    "LEVEL1_NAMES",
    "NONDRIVABLE",
    "DRIVABLE",
    "IGNORE_INDEX",
    "load_label_table",
    "drivable_names",
    "build_lut",
    "max_id_for_level",
    "detect_level",
    "remap_mask",
    "polygons_to_mask",
]
