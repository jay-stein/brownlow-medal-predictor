from brownlow.naming import (
    compact_key,
    make_name_key,
    normalise_name,
    normalise_team,
    strip_name_suffix,
)


def test_normalise_name_strips_accents_and_punctuation():
    assert normalise_name("Orazio Fantasia") == "ORAZIO FANTASIA"
    assert normalise_name("Connor O'Brien") == "CONNOR O BRIEN"
    assert normalise_name("Luke Davies-Uniacke") == "LUKE DAVIES UNIACKE"
    assert normalise_name("  Text  ") == "TEXT"


def test_compact_key_ignores_punctuation_and_spacing():
    assert compact_key("Connor O'Brien") == "CONNOROBRIEN"
    assert compact_key("Massimo D'Ambrosio") == "MASSIMODAMBROSIO"
    assert compact_key("Luke Davies-Uniacke") == "LUKEDAVIESUNIACKE"


def test_normalise_team_aliases():
    assert normalise_team("Brisbane") == normalise_team("Brisbane Lions")
    assert normalise_team("GWS") == normalise_team("Greater Western Sydney")
    assert normalise_team("Sydney") == normalise_team("Sydney Swans")
    assert normalise_team("West Coast") == normalise_team("West Coast Eagles")
    assert normalise_team("North Melbourne") == normalise_team("North Melbourne")


def test_make_name_key():
    assert make_name_key(None, "Smith") == "SMITH"
    assert make_name_key("Nick", "Daicos") == "NICKDAICOS"
    assert make_name_key("Cam", "O'Shea") == make_name_key("Cam", "OShea")


def test_strip_name_suffix():
    assert strip_name_suffix(compact_key("Alwyn Davey Jnr")) == "ALWYNDAVEY"
    assert strip_name_suffix(compact_key("Robert Hansen Jr")) == "ROBERTHANSEN"
    assert strip_name_suffix(compact_key("Sam Reid")) == "SAMREID"
