class Redactor:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.reverse_mapping: dict[str, str] = {}
        for original, replacement in mapping.items():
            self.reverse_mapping[replacement] = "Kotak"

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
