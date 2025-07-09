from collections.abc import Callable, Sequence

from app.schemas.configuration import OutputConfig
from app.schemas.configuration.output_config import DestinationType
from app.services.dispatchers import Dispatcher, FolderDispatcher, MqttDispatcher


class DispatchService:
    _dispatcher_registry: dict[DestinationType, Callable[[OutputConfig], Dispatcher | None]] = {
        DestinationType.DISCONNECTED: lambda _: None,
        DestinationType.FOLDER: lambda config: FolderDispatcher(output_config=config),
        DestinationType.MQTT: lambda config: MqttDispatcher(output_config=config),
        DestinationType.DDS: lambda _: _raise_not_implemented("DDS output is not implemented yet"),
        DestinationType.ROS: lambda _: _raise_not_implemented("ROS output is not implemented yet"),
    }

    @classmethod
    def get_destination(cls, output_config: OutputConfig) -> Dispatcher | None:
        # TODO handle exceptions: if some output cannot be initialized, exclude it and raise a warning
        factory = cls._dispatcher_registry.get(output_config.destination_type)
        if factory is None:
            raise ValueError(f"Unrecognized destination type: {output_config.destination_type}")

        return factory(output_config)

    @classmethod
    def get_destinations(cls, output_configs: Sequence[OutputConfig]) -> list[Dispatcher]:
        """
        Get a list of dispatchers based on the provided output configurations.

        Args:
            output_configs (Sequence[OutputConfig]): A sequence of output configurations.
        """
        return [dispatcher for config in output_configs if (dispatcher := cls.get_destination(config)) is not None]


def _raise_not_implemented(message: str) -> None:
    raise NotImplementedError(message)
