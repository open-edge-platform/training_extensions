import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import anyio
from fastapi import UploadFile
from model_api.models import DetectionModel

from app.schemas.model_activation import ModelActivationState
from app.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class ModelAlreadyExistsError(Exception):
    """Exception raised when a model with the same name already exists"""


class ModelNotFoundError(Exception):
    """Exception raised when a model is not found"""


@dataclass
class LoadedModel:
    name: str
    model: DetectionModel


class ModelService(metaclass=Singleton):
    """Service to register and activate models"""

    def __init__(self) -> None:
        self.models_dir = Path("data/models")
        self.state_file = Path("data/models_state.json")

        self._model_activation_state: ModelActivationState = self._load_state()
        self._model_activation_state_lock = Lock()

        self._loaded_model: LoadedModel | None = None

    def _load_state(self) -> ModelActivationState:
        """Load the state from the file if it exists, otherwise initialize an empty state"""
        if self.state_file.exists():
            try:
                return ModelActivationState.from_json_dict(json.load(self.state_file.open()))
            except json.JSONDecodeError as e:
                logger.error(f"Error loading models state from {self.state_file}: {e}")
        return ModelActivationState(active_model=None, available_models=[])

    def _save_state(self) -> None:
        """Save the state to the file"""
        with open(self.state_file, "w") as f:
            json.dump(self._model_activation_state.to_json_dict(), f)

    def _get_model_xml_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.xml"

    def _get_model_bin_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.bin"

    async def add_model(self, model_name: str, model_xml_file: UploadFile, model_bin_file: UploadFile) -> None:
        """
        Store a new model and make it available for inference

        Args:
            model_name: Name of the model
            model_xml_file: XML file describing the model topology
            model_bin_file: BIN file containing the model weights
        """
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

        xml_path = self._get_model_xml_path(model_name)
        bin_path = self._get_model_bin_path(model_name)

        # If a model is already registered with the same name, raise an error
        if model_name in self._model_activation_state.available_models:
            raise ModelAlreadyExistsError(f"A model with the name '{model_name}' already exists")

        with self._model_activation_state_lock:
            # Save the files
            async with await anyio.open_file(xml_path, "wb") as f:
                while chunk := await model_xml_file.read(1024 * 1024):  # 1MB chunks
                    await f.write(chunk)
            async with await anyio.open_file(bin_path, "wb") as f:
                while chunk := await model_bin_file.read(1024 * 1024):  # 1MB chunks
                    await f.write(chunk)

            # Add the model to the inference state
            self._model_activation_state.available_models.append(model_name)

            # Activate the model if it is the first model to be added
            if self._model_activation_state.active_model is None:
                self._model_activation_state.active_model = model_name

            # Store the state
            self._save_state()

    def remove_model(self, model_name: str) -> None:
        """
        Remove a previously added model

        Args:
            model_name: Name of the model to remove
        """
        with self._model_activation_state_lock:
            # If the model does not exist, raise an error
            if model_name not in self._model_activation_state.available_models:
                raise ModelNotFoundError(f"Model '{model_name}' not found")

            # Remove the model from the inference state
            self._model_activation_state.available_models.remove(model_name)

            # If the model is active, deactivate it and activate the next available model
            if self._model_activation_state.active_model == model_name:
                logger.info(f"Deactivating model '{model_name}'")
                if len(self._model_activation_state.available_models) > 0:
                    next_model = self._model_activation_state.available_models[0]
                    logger.info(f"Activating next available model '{next_model}'")
                    self._model_activation_state.active_model = next_model
                else:
                    logger.info("No more available models to activate")
                    self._model_activation_state.active_model = None

            # Remove the files
            xml_path = self.models_dir / f"{model_name}.xml"
            bin_path = self.models_dir / f"{model_name}.bin"
            xml_path.unlink()
            bin_path.unlink()

            # Store the state
            self._save_state()

    def get_available_model_names(self) -> list[str]:
        """Get the names of all available models"""
        return self._model_activation_state.available_models

    def get_active_model_name(self) -> str | None:
        """Get the name of the active model"""
        return self._model_activation_state.active_model

    def activate_model(self, model_name: str) -> None:
        """Activate a model for inference"""
        logger.info(f"Activating model '{model_name}'")
        with self._model_activation_state_lock:
            # If there is no model available with the given name, raise an error
            if model_name not in self._model_activation_state.available_models:
                raise ModelNotFoundError(f"Model '{model_name}' not found")

            # Activate the model
            self._model_activation_state.active_model = model_name

            # Store the state
            self._save_state()

    def get_inference_model(self) -> DetectionModel | None:
        if self._model_activation_state.active_model is None:
            return None

        if self._loaded_model is None or self._loaded_model.name != self._model_activation_state.active_model:
            logger.info(f"Loading model '{self._model_activation_state.active_model}'")
            model_path = self._get_model_xml_path(self._model_activation_state.active_model)
            self._loaded_model = LoadedModel(
                name=self._model_activation_state.active_model,
                model=DetectionModel.create_model(model_path),
            )
        return self._loaded_model.model
