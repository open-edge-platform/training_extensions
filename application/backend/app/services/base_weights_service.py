# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import shutil
from pathlib import Path

import requests

from app.supported_models import ModelManifest
from app.supported_models.default_models import TaskType
from app.supported_models.supported_models import SupportedModels

logger = logging.getLogger(__name__)


class BaseWeightsService:
    """Service for downloading and managing pretrained model weights from external archives."""

    def __init__(self, data_dir: Path) -> None:
        self.pretrained_weights_dir = data_dir / "pretrained_weights"
        for task in TaskType:
            task_dir = self.pretrained_weights_dir / task.name.lower()
            task_dir.mkdir(parents=True, exist_ok=True)

    def get_remote_weights_path(self, task: TaskType, model_manifest_id: str) -> str:
        """
        Return the remote location of the weights as configured in the manifest.

        Args:
            task: The task type (used for validation)
            model_manifest_id: The unique identifier of the model architecture

        Returns:
            str: The remote URL of the pretrained weights
        """
        manifest = self._get_and_validate_model_manifest(task, model_manifest_id)
        return manifest.pretrained_weights.url

    def get_local_weights_path(self, task: TaskType, model_manifest_id: str, allow_download: bool = True) -> Path:
        """
        Return the location of the weights (.pt file) in local storage.

        If not already present and allow_download is enabled, downloads the weights from remote.

        Args:
            task: The task type
            model_manifest_id: The unique identifier of the model architecture
            allow_download: Whether to download weights if not present locally

        Returns:
            Path: Path to the local weights file

        Raises:
            FileNotFoundError: If weights are not found locally and allow_download is False
        """
        manifest = self._get_and_validate_model_manifest(task, model_manifest_id)

        local_path = self.pretrained_weights_dir / task.name.lower() / Path(manifest.pretrained_weights.url).name
        if local_path.exists():
            if self._verify_file_integrity(file_path=local_path, sha_sum=manifest.pretrained_weights.sha_sum):
                logger.info(f"Using cached weights for {model_manifest_id}: {local_path}")
                return local_path

            logger.warning(f"Cached weights for {model_manifest_id} failed integrity check, will re-download")
            local_path.unlink()

        if not allow_download:
            raise FileNotFoundError(f"Weights not found locally for model {model_manifest_id} and download is disabled")

        logger.info(f"Downloading pretrained weights for {model_manifest_id} from {manifest.pretrained_weights.url}")
        self._download_weights(
            remote_url=manifest.pretrained_weights.url,
            local_path=local_path,
            sha_sum=manifest.pretrained_weights.sha_sum,
        )

        return local_path

    def remove_local_weights(self, task: TaskType, model_manifest_id: str) -> bool:
        """
        Delete the local base weights of a specific model architecture to free space on disk.

        Args:
            task: The task type
            model_manifest_id: The unique identifier of the model architecture

        Returns:
            bool: True if weights were successfully removed, False if they didn't exist
        """
        manifest = self._get_and_validate_model_manifest(task, model_manifest_id)
        local_path = self.pretrained_weights_dir / task.name.lower() / Path(manifest.pretrained_weights.url).name
        if local_path.exists():
            try:
                local_path.unlink()
                logger.info(f"Removed local weights for {model_manifest_id}: {local_path}")
                return True
            except OSError as e:
                logger.error(f"Failed to remove weights file {local_path}: {e}")

        return False

    def remove_all_local_weights(self) -> int:
        """
        Remove all locally cached pretrained weights to free space on disk.

        Returns:
            int: Number of weight files that were removed
        """
        removed_count = 0
        if not self.pretrained_weights_dir.exists():
            return 0

        try:
            for weights_file in self.pretrained_weights_dir.rglob("*"):
                if weights_file.is_file() and not weights_file.name.startswith("."):
                    try:
                        weights_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed weights file: {weights_file}")
                    except OSError as e:
                        logger.error(f"Failed to remove weights file {weights_file}: {e}")

            logger.info(f"Removed {removed_count} cached weight files")
            return removed_count
        except OSError as e:
            logger.error(f"Failed to remove cached weights: {e}")
            return 0

    @staticmethod
    def _verify_file_integrity(file_path: Path, sha_sum: str) -> bool:
        """
        Verify the integrity of a downloaded file using SHA256 checksum.

        Args:
            file_path: Path to the file to verify
            sha_sum: Expected SHA256 checksum

        Returns:
            bool: True if the file integrity is valid, False otherwise
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            actual_sha_sum = sha256_hash.hexdigest()
            return actual_sha_sum == sha_sum
        except Exception as e:
            logger.error(f"Failed to verify file integrity for {file_path}: {e}")
            return False

    def _check_disk_space(self, remote_url: str, safety_margin_gb: float = 1.0) -> None:
        """
        Check if there's sufficient disk space for downloading the remote file.

        Args:
            remote_url: URL of the file to download
            safety_margin_gb: Additional safety margin in GB

        Raises:
            OSError: If there's insufficient disk space
        """
        try:
            # Use a longer read timeout to allow for large file downloads
            response = requests.head(remote_url, allow_redirects=True, timeout=(10, 600))
            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length:
                file_size = int(content_length)
            else:
                # If we can't get the size, assume a reasonable default (500MB)
                file_size = 500 * 1024 * 1024
                logger.warning(f"Could not determine file size for {remote_url}, assuming 500MB")
        except Exception as e:
            logger.warning(f"Could not check remote file size for {remote_url}: {e}, assuming 500MB")
            file_size = 500 * 1024 * 1024

        stat = shutil.disk_usage(self.pretrained_weights_dir)
        available_space = stat.free
        required_space = file_size + (safety_margin_gb * 1024 * 1024 * 1024)

        if available_space < required_space:
            raise OSError(
                f"Insufficient disk space. Required: {required_space / (1024**3):.2f} GB, "
                f"Available: {available_space / (1024**3):.2f} GB"
            )

    def _download_weights(self, remote_url: str, local_path: Path, sha_sum: str) -> None:
        """
        Download weights from remote URL to local path and verify integrity.
        Before download, checks if there is enough space on disk.

        Args:
            remote_url: URL to download from
            local_path: Local path to save the file
            sha_sum: Expected SHA256 checksum for verification

        Raises:
            ValueError: If downloaded file fails integrity check, or if download fails
        """
        self._check_disk_space(remote_url)

        # Create temporary file for download
        temp_path = local_path.with_suffix(".tmp")
        try:
            # Stream the download to handle large files, use a longer read timeout to allow for large file downloads
            with requests.get(remote_url, stream=True, timeout=(10, 600)) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    for data in response.iter_content(chunk_size=4096):
                        f.write(data)

            if not self._verify_file_integrity(temp_path, sha_sum):
                temp_path.unlink()
                raise ValueError(f"Downloaded file failed integrity check for {remote_url}")

            # Move temporary file to final location
            temp_path.rename(local_path)
            logger.info(f"Successfully downloaded and verified weights: {local_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to download weights from {remote_url}: {e}")
            raise ValueError(f"Failed to download weights from {remote_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error downloading weights from {remote_url}: {e}")
            raise ValueError(f"Unexpected error downloading weights from {remote_url}: {e}")
        finally:
            # Clean up temporary file if it exists
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _get_and_validate_model_manifest(task: TaskType, model_manifest_id: str) -> ModelManifest:
        """
        Validate or retrieve the default model manifest ID for a given task.

        Args:
            task: The task type (e.g., classification, detection)
            model_manifest_id: The provided model manifest ID, or None to use default

        Returns:
            ModelManifest: The validated model manifest

        Raises:
            ValueError: If the model manifest is not found, task type mismatch, or doesn't have pretrained weights
        """
        manifest = SupportedModels.get_model_manifest_by_id(model_manifest_id)
        if manifest.task != task:
            raise ValueError(f"Task mismatch: expected '{task.name.lower()}', got '{manifest.task.lower()}'")
        return manifest
