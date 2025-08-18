from threading import Lock


class Singleton(type):
    _instance = None
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:  # double-checked locking
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
