import multiprocessing as mp
import queue

FRAME_QUEUE_SIZE = 5
PREDICTION_QUEUE_SIZE = 5

# Queue for the frames acquired from the stream source and decoded
frame_queue: mp.Queue = mp.Queue(maxsize=FRAME_QUEUE_SIZE)

# Queue for the inference results (predictions)
pred_queue: mp.Queue = mp.Queue(maxsize=PREDICTION_QUEUE_SIZE)

# Queue for pushing predictions to the visualization stream (WebRTC)
rtc_stream_queue: queue.Queue = queue.Queue(maxsize=1)

# Event to sync all processes on application shutdown
mp_stop_event = mp.Event()

# Event to signal that the model has to be reloaded
mp_reload_model_event = mp.Event()

# Condition variable to notify processes about configuration updates
mp_config_changed_condition = mp.Condition()
