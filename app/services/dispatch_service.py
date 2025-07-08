from collections.abc import Sequence

from app.entities.dispatchers import Dispatcher, FolderDispatcher
from app.schemas.configuration import OutputConfig
from app.schemas.configuration.output_config import DestinationType


class DispatchService:
    @staticmethod
    def get_destination(output_config: OutputConfig) -> Dispatcher | None:
        # TODO handle exceptions: if some output cannot be initialized, exclude it and raise a warning
        match output_config.destination_type:
            case DestinationType.DISCONNECTED:
                return None
            case DestinationType.FOLDER:
                return FolderDispatcher(output_config=output_config)
            case DestinationType.MQTT:
                raise NotImplementedError("MQTT output is not implemented yet")
            case DestinationType.DDS:
                raise NotImplementedError("DDS output is not implemented yet")
            case DestinationType.ROS:
                raise NotImplementedError("ROS output is not implemented yet")
            case _:
                raise ValueError(f"Unrecognized destination type: {output_config.destination_type}")

    @staticmethod
    def get_destinations(output_configs: Sequence[OutputConfig]) -> list[Dispatcher]:
        """
        Get a list of dispatchers based on the provided output configurations.

        Args:
            output_configs (Sequence[OutputConfig]): A sequence of output configurations.
        """
        dispatchers: list[Dispatcher] = []
        for config in output_configs:
            dispatcher = DispatchService.get_destination(config)
            if dispatcher is not None:
                dispatchers.append(dispatcher)
        return dispatchers
