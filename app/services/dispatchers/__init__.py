from .filesystem import FolderDispatcher
from .mqtt import MqttDispatcher

Dispatcher = FolderDispatcher

__all__ = ["Dispatcher", "FolderDispatcher", "MqttDispatcher"]
