# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import MagicMock, call, patch

from app.entities.images_folder_stream import ImagesFolderStream


class TestImagesFolderStream:
    """Test cases for ImagesFolderStream."""

    @patch.object(ImagesFolderStream, "_init_watchdog")
    @patch("os.path.getmtime", return_value=time.time())
    @patch("os.path.isfile", return_value=True)
    @patch("os.listdir", return_value=["file1", "file2"])
    def test_init_ignore_existing_images(self, mock_listdir, mock_isfile, mock_getmtime, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=True)
        assert stream.files == []
        mock_listdir.assert_not_called()
        mock_isfile.assert_not_called()
        mock_getmtime.assert_not_called()
        mock_init_watchdog.assert_called_once_with("folder_path")

    @patch.object(ImagesFolderStream, "_init_watchdog")
    @patch("os.path.getmtime", return_value=time.time())
    @patch("os.path.isfile", return_value=True)
    @patch("os.listdir", return_value=["file1", "file2"])
    def test_init_do_not_ignore_existing_images(self, mock_listdir, mock_isfile, mock_getmtime, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=False)
        assert stream.files == ["folder_path/file1", "folder_path/file2"]
        mock_listdir.assert_called_once_with("folder_path")
        mock_isfile.assert_has_calls([call("folder_path/file1"), call("folder_path/file2")])
        mock_getmtime.assert_has_calls([call("folder_path/file1"), call("folder_path/file2")])
        mock_init_watchdog.assert_called_once_with("folder_path")

    @patch.object(ImagesFolderStream, "_init_watchdog")
    def test_get_data_empty_list(self, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=True)
        data = stream.get_data()
        assert data is None

    @patch.object(ImagesFolderStream, "_init_watchdog")
    def test_get_data_cannot_load_image(self, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=True)
        stream.files = ["folder_path/file1", "folder_path/file2"]
        data = stream.get_data()
        assert data is None

    @patch.object(ImagesFolderStream, "_init_watchdog")
    @patch("os.path.getmtime", return_value=time.time())
    @patch("cv2.imread", return_value=MagicMock())
    def test_get_data(self, mock_imread, mock_getmtime, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=True)
        stream.files = ["folder_path/file1", "folder_path/file2"]
        data = stream.get_data()
        assert data is not None
        mock_imread.assert_called_once_with("folder_path/file1")
        mock_getmtime.assert_called_once_with("folder_path/file1")

    @patch.object(ImagesFolderStream, "_init_watchdog")
    def test_file_added(self, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=True)
        stream.files = ["folder_path/file1", "folder_path/file2"]
        stream.file_added("folder_path/file3")
        assert stream.files == ["folder_path/file1", "folder_path/file2", "folder_path/file3"]

    @patch.object(ImagesFolderStream, "_init_watchdog")
    def test_file_deleted(self, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=True)
        stream.files = ["folder_path/file1", "folder_path/file2"]
        stream.file_deleted("folder_path/file1")
        assert stream.files == ["folder_path/file2"]

    @patch.object(ImagesFolderStream, "_init_watchdog")
    def test_file_deleted_not_in_list(self, mock_init_watchdog):
        stream = ImagesFolderStream(folder_path="folder_path", ignore_existing_images=True)
        stream.files = ["folder_path/file1", "folder_path/file2"]
        stream.file_deleted("folder_path/file3")
        assert stream.files == ["folder_path/file1", "folder_path/file2"]
