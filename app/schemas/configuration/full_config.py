from pydantic import BaseModel, ConfigDict

from .input_config import InputConfig
from .output_config import OutputConfig

### Full config model


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: InputConfig
    outputs: list[OutputConfig]
