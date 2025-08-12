# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py  (UI_MODE=True)
#  - uv run fastapi run  (UI_MODE=False)
#  - uv run fastapi dev  (UI_MODE=False, development mode)
# or use docker and access UI and backend at geti-edge.localhost
#  - docker compose up
#  - docker compose -f docker-compose.dev.yaml up

import logging
from collections.abc import Iterator
from pathlib import Path

import anyio
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastrtc import AdditionalOutputs, Stream
from pydantic import BaseModel, Field

from app.api.endpoints import configuration, models, pipelines, sinks, sources, system
from app.core import Scheduler, lifespan
from app.settings import get_settings
from app.utils.ipc import mp_stop_event

settings = get_settings()

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def rtc_stream_routine() -> Iterator[tuple[np.ndarray, AdditionalOutputs]]:
    """Iterator to send frames with predictions to the WebRTC visualization stream"""
    while not mp_stop_event.is_set():
        yield Scheduler().rtc_stream_queue.get()
    logger.info("Stopped RTC stream routine")


stream = Stream(
    handler=rtc_stream_routine,
    modality="video",
    mode="receive",
    additional_outputs=[
        gr.Textbox(label="Predictions"),
    ],
    additional_outputs_handler=lambda _c1, pred: pred,
)

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    openapi_url=settings.openapi_url,
    redoc_url=None,
    docs_url=None,
    lifespan=lifespan,
    # TODO add contact info
    # TODO add license
)

app.add_middleware(  # TODO restrict settings in production
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sources.router)
app.include_router(sinks.router)
app.include_router(pipelines.router)
app.include_router(models.router)
app.include_router(configuration.router)
app.include_router(system.router)

cur_dir = Path(__file__).parent


@app.get("/api/docs", include_in_schema=False)
async def get_scalar_docs() -> HTMLResponse:
    """Shows docs for our OpenAPI specification using scalar"""
    async with await anyio.open_file(cur_dir / "scalar.html") as file:
        html_content = await file.read()
        return HTMLResponse(content=html_content)


class InputData(BaseModel):
    webrtc_id: str
    conf_threshold: float = Field(ge=0, le=1)


# TODO remove this endpoint, make sure the UI does not require it
@app.post("/api/input_hook", tags=["webrtc"])
async def webrtc_input_hook(data: InputData) -> None:
    """Update webrtc input for user"""
    stream.set_input(data.webrtc_id, data.conf_threshold)


stream.mount(app, "/api")


def main() -> None:
    """Main application entry point"""
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")

    if settings.gradio_ui:
        logger.info("Starting Gradio UI...")
        stream.ui.launch(server_name=settings.host, server_port=settings.port, show_error=settings.debug)
    else:
        logger.info("Starting FastAPI server...")

        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.debug and settings.environment == "dev",
            log_level="debug" if settings.debug else "info",
        )


if __name__ == "__main__":
    main()
