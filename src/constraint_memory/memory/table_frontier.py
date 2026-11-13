"""E3T: the E3F retrieval policy with reversible exact binding tables."""
from .binding_tables import serialize_sections
from .frontier import FrontierMemory


class TableFrontierMemory(FrontierMemory):
    def __init__(self, config=None):
        super().__init__(config)
        self.variant = "E3T"
        self.features["lossless_binding_tables"] = True

    @staticmethod
    def _serialize(sections):
        return serialize_sections(sections)
