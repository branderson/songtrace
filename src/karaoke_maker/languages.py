from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    iso: str             # whisper / num2words language code
    pyphen: str | None   # pyphen dictionary code, None if no dictionary exists


LANGUAGES: dict[str, Language] = {
    "english": Language("en", "en_US"),
    "german": Language("de", "de_DE"),
    "spanish": Language("es", "es"),
    "french": Language("fr", "fr"),
    "italian": Language("it", "it"),
    "portuguese": Language("pt", "pt"),
    "dutch": Language("nl", "nl_NL"),
    "japanese": Language("ja", None),
}
