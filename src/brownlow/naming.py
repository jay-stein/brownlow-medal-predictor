"""Name and team normalisation helpers.

The Champion Data (fitzRoy ``fetch_player_stats``) files and the AFL Tables
(``fetch_player_stats_afltables``) files use different player id namespaces and
different team name conventions. These helpers produce comparable keys.
"""

from __future__ import annotations

import re
import unicodedata

_TEAM_ALIASES = {
    "ADELAIDE": "ADELAIDE CROWS",
    "ADELAIDECROWS": "ADELAIDE CROWS",
    "BRISBANE": "BRISBANE LIONS",
    "BRISBANELIONS": "BRISBANE LIONS",
    "CARLTON": "CARLTON",
    "COLLINGWOOD": "COLLINGWOOD",
    "ESSENDON": "ESSENDON",
    "FREMANTLE": "FREMANTLE",
    "GEELONG": "GEELONG CATS",
    "GEELONGCATS": "GEELONG CATS",
    "GOLDCOAST": "GOLD COAST SUNS",
    "GOLDCOASTSUNS": "GOLD COAST SUNS",
    "GREATERWESTERNSYDNEY": "GWS GIANTS",
    "GWS": "GWS GIANTS",
    "GWSGIANTS": "GWS GIANTS",
    "HAWTHORN": "HAWTHORN",
    "MELBOURNE": "MELBOURNE",
    "NORTHMELBOURNE": "NORTH MELBOURNE",
    "PORTADELAIDE": "PORT ADELAIDE",
    "RICHMOND": "RICHMOND",
    "STKILDA": "ST KILDA",
    "SYDNEY": "SYDNEY SWANS",
    "SYDNEYSWANS": "SYDNEY SWANS",
    "WESTCOAST": "WEST COAST EAGLES",
    "WESTCOASTEAGLES": "WEST COAST EAGLES",
    "WESTERNBULLDOGS": "WESTERN BULLDOGS",
}

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_NAME_SUFFIX = re.compile(r"(JNR|JR|SR|II|III|IV)$")


def normalise_name(value: str) -> str:
    """Upper-case, strip accents/punctuation and collapse whitespace."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper()
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def compact_key(value: str | None) -> str:
    """Punctuation- and space-insensitive key (``O'Shea`` == ``OShea``)."""
    return normalise_name(value).replace(" ", "")


def strip_name_suffix(key: str) -> str:
    """Remove generational suffixes (Jr/Jnr/Sr/II-IV) from a compact key."""
    return _NAME_SUFFIX.sub("", key)


def normalise_team(value: str) -> str:
    """Map a team name (any source) to a canonical key."""
    key = normalise_name(value).replace(" ", "")
    return _TEAM_ALIASES.get(key, normalise_name(value))


def make_name_key(first: str | None, last: str | None) -> str:
    """Build a compact full-name key from first and last name parts."""
    return compact_key(f"{first or ''} {last or ''}")
