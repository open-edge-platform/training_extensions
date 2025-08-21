from enum import StrEnum


class Tags(StrEnum):
    """Enum for API tags."""

    MODELS = "Models"
    SINKS = "Sinks"
    SOURCES = "Sources"
    PIPELINES = "Pipelines"
    ANNOTATIONS = "Annotations"
