"""Text redaction — replaces sensitive terms before they reach the LLM."""


class Redactor:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        # Build reverse mapping: replacement → first original form encountered.
        # When multiple originals map to the same replacement (Kotak/kotak/KOTAK → KKK)
        # the first one is used for unmasking.
        self.reverse_mapping: dict[str, str] = {}
        for original, replacement in mapping.items():
            if replacement not in self.reverse_mapping:
                self.reverse_mapping[replacement] = original

    def mask_text(self, text: str) -> str:
        masked = text
        for original, replacement in self.mapping.items():
            masked = masked.replace(original, replacement)
        return masked

    def unmask_text(self, text: str) -> str:
        unmasked = text
        for replacement, original in self.reverse_mapping.items():
            unmasked = unmasked.replace(replacement, original)
        return unmasked
