import psutil

from utils.singleton import Singleton


class SystemService(metaclass=Singleton):
    """Service to get system information"""
    def __init__(self) -> None:
        self.process = psutil.Process()


    def get_memory_usage(self):
        """
        Get the memory usage of the process

        Returns:
            tuple[float, float]: Memory usage in MB and the relative percentage
        """
        return self.process.memory_info().rss / (1024 * 1024), self.process.memory_percent()

    def get_cpu_usage(self) -> float:
        """
        Get the CPU usage of the process

        Returns:
            float: CPU usage in percentage
        """
        return self.process.cpu_percent(interval=None)
