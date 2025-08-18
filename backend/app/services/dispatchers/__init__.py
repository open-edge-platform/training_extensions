from .filesystem import FolderDispatcher
from .mqtt import MqttDispatcher

Dispatcher = FolderDispatcher | MqttDispatcher

__all__ = ["Dispatcher", "FolderDispatcher", "MqttDispatcher"]
