import multiprocessing as mp

# Event to sync all processes on application shutdown
mp_stop_event = mp.Event()

# Event to signal that the model has to be reloaded
mp_reload_model_event = mp.Event()

# Condition variable to notify processes about configuration updates
mp_config_changed_condition = mp.Condition()
