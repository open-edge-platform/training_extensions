from pydantic import BaseModel, ConfigDict, Field

from .input_config import DisconnectedSourceConfig, Source
from .output_config import DisconnectedOutputConfig, Sink

### Full config model


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Source = Field(default_factory=DisconnectedSourceConfig)
    output: Sink = Field(default_factory=lambda: DisconnectedOutputConfig(output_formats=[]))
