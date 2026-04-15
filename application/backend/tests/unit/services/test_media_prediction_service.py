# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import ANY, MagicMock, call, patch
from uuid import uuid4

import numpy as np
import PIL.Image as PILImage
import pytest
from sqlalchemy.orm import Session

from app.models import (
    BatchInferenceInput,
    BatchInferenceMedia,
    BatchInferencePrediction,
    BatchInferenceResult,
    DatasetItem,
    DatasetItemAnnotation,
    Project,
    Task,
    Video,
)
from app.models.media import (
    Image,
    MediaListPredictionRequest,
    MediaPredictionRequest,
    MediaType,
    NotAnnotatedVideoFrame,
    VideoFrame,
    VideoRange,
)
from app.models.system import DeviceInfo, DeviceType
from app.services import (
    DatasetService,
    LabelService,
    MediaPredictionService,
    MediaService,
    ResourceNotFoundError,
    ResourceType,
)
from app.services.inference import InferenceServer
from app.services.media_prediction_service import LoadedMedia


class TestMediaPredictionServiceUnit:
    """Unit tests for MediaPredictionService."""

    @pytest.fixture
    def fxt_label_service(self):
        return MagicMock(spec=LabelService)

    @pytest.fixture
    def fxt_media_service(self):
        return MagicMock(spec=MediaService)

    @pytest.fixture
    def fxt_dataset_service(self):
        return MagicMock(spec=DatasetService)

    @pytest.fixture
    def fxt_inference_server(self):
        return MagicMock(spec=InferenceServer)

    @pytest.fixture
    def fxt_media_prediction_service(
        self, fxt_label_service, fxt_media_service, fxt_dataset_service, fxt_inference_server
    ):
        db_session = MagicMock(spec=Session)
        return MediaPredictionService(
            label_service=fxt_label_service,
            media_service=fxt_media_service,
            dataset_service=fxt_dataset_service,
            inference_server=fxt_inference_server,
            inference_model_ttl=10,
            db_session=db_session,
        )

    def test_load_frame_range(self, fxt_media_prediction_service, fxt_media_service):
        project = MagicMock(spec=Project)
        video = MagicMock(spec=Video, id=uuid4())

        # One frame is already annotated
        video_frame = MagicMock(spec=VideoFrame, video_id=video.id, frame_index=0)
        fxt_media_service.search_video_frames_by_video_id_and_indexes.return_value = [video_frame]

        result = fxt_media_prediction_service._load_frame_range(
            project=project, video=video, video_range=VideoRange(start_frame=0, end_frame=2, stride=1)
        )

        # Assert that one annotated and one unannotated frame is returned
        fxt_media_service.search_video_frames_by_video_id_and_indexes.assert_called_once_with(
            project=project, video_id=video.id, frame_indexes=[0, 1, 2]
        )
        assert result == [
            video_frame,
            NotAnnotatedVideoFrame(video=video, frame_index=1),
            NotAnnotatedVideoFrame(video=video, frame_index=2),
        ]

    def test_load_media(self, fxt_media_prediction_service, fxt_media_service):
        model_id = uuid4()
        image_id = uuid4()
        video_id = uuid4()
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=image_id, type=MediaType.IMAGE)
        video = MagicMock(spec=Video, id=video_id, type=MediaType.VIDEO)

        video_range = VideoRange(start_frame=0, end_frame=1, stride=1)
        fxt_media_service.get_media_by_ids.return_value = [image, video]

        with patch.object(
            fxt_media_prediction_service,
            "_load_frame_range",
            return_value=[NotAnnotatedVideoFrame(video=video, frame_index=0)],
        ) as mock_load_frame_range:
            result = fxt_media_prediction_service._load_media(
                project=project,
                request=MediaListPredictionRequest(
                    model_id=model_id,
                    media=[
                        MediaPredictionRequest(media_id=image_id, range=None),
                        MediaPredictionRequest(media_id=video_id, range=video_range),
                    ],
                    save_predictions=False,
                    device="AUTO",
                ),
            )

        fxt_media_service.get_media_by_ids.assert_called_once_with(
            project_id=project.id, media_ids=[image_id, video_id]
        )
        mock_load_frame_range.assert_called_once_with(project=project, video=video, video_range=video_range)
        assert result == LoadedMedia(
            single_media=[image],
            video_frames=[NotAnnotatedVideoFrame(video=video, frame_index=0)],
        )

    def test_load_media_not_found(self, fxt_media_prediction_service, fxt_media_service):
        model_id = uuid4()
        media_id = uuid4()
        project = MagicMock(spec=Project, id=uuid4())

        fxt_media_service.get_media_by_ids.return_value = []

        with pytest.raises(ResourceNotFoundError):
            fxt_media_prediction_service._load_media(
                project=project,
                request=MediaListPredictionRequest(
                    model_id=model_id,
                    media=[
                        MediaPredictionRequest(media_id=media_id, range=None),
                    ],
                    save_predictions=False,
                    device="AUTO",
                ),
            )

        fxt_media_service.get_media_by_ids.assert_called_once_with(project_id=project.id, media_ids=[media_id])

    def test_convert_to_inference_input(self, fxt_media_prediction_service, fxt_media_service):
        image_id = uuid4()
        video_id = uuid4()
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=image_id, type=MediaType.IMAGE)
        video = MagicMock(spec=Video, id=video_id, type=MediaType.VIDEO)
        video_frame = MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, video_id=video_id, frame_index=0)

        not_annotated_video_frame = NotAnnotatedVideoFrame(video=video, frame_index=1)
        not_annotated_video_frame_binary = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)

        loaded_media = LoadedMedia(
            single_media=[image],
            video_frames=[video_frame, not_annotated_video_frame],
        )

        fxt_media_service.get_frame_binaries.return_value = {1: not_annotated_video_frame_binary}

        with patch.object(fxt_media_prediction_service, "_load_media_binary") as mock_load_media_binary:
            mock_load_media_binary.side_effect = [np.random.rand(100, 100, 3), np.random.rand(100, 100, 3)]
            result_inputs = fxt_media_prediction_service._convert_to_inference_input(
                project=project,
                loaded_media=loaded_media,
            )

        mock_load_media_binary.assert_has_calls(
            [
                call(project_id=project.id, media=image),
                call(project_id=project.id, media=video_frame),
            ]
        )
        fxt_media_service.get_frame_binaries.assert_called_once_with(project=project, video=video, frame_indexes=[1])
        assert len(result_inputs) == 3
        na_frame_input = next(inp for inp in result_inputs if inp.media_id == video_id and inp.frame_index == 1)
        np.testing.assert_array_equal(na_frame_input.data, not_annotated_video_frame_binary)

    def test_create_or_update_dataset_item_not_found(self, fxt_media_prediction_service, fxt_dataset_service):
        model_id = uuid4()
        task = MagicMock(spec=Task)
        project = MagicMock(spec=Project, id=uuid4(), task=task)
        image = MagicMock(spec=Image, id=uuid4(), type=MediaType.IMAGE)
        annotation = MagicMock(spec=DatasetItemAnnotation)
        prediction = MagicMock(spec=BatchInferencePrediction, prediction=[annotation])

        fxt_dataset_service.get_dataset_item_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, image.id
        )

        fxt_media_prediction_service._create_or_update_dataset_item(
            project=project,
            media=image,
            prediction=prediction,
            model_id=model_id,
        )

        fxt_dataset_service.create_dataset_item.assert_called_once_with(
            project_id=project.id,
            task=task,
            media=image,
            annotations=[annotation],
            user_reviewed=False,
            prediction_model_id=model_id,
        )

    def test_create_or_update_dataset_item_not_reviewed(self, fxt_media_prediction_service, fxt_dataset_service):
        model_id = uuid4()
        dataset_item_id = uuid4()
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=uuid4(), type=MediaType.IMAGE)
        dataset_item = MagicMock(spec=DatasetItem, id=dataset_item_id, user_reviewed=False)
        annotation = MagicMock(spec=DatasetItemAnnotation)
        prediction = MagicMock(spec=BatchInferencePrediction, prediction=[annotation])

        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        fxt_media_prediction_service._create_or_update_dataset_item(
            project=project,
            media=image,
            prediction=prediction,
            model_id=model_id,
        )

        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=project,
            dataset_item_id=dataset_item_id,
            annotations=[annotation],
            user_reviewed=False,
            prediction_model_id=model_id,
        )

    def test_create_or_update_dataset_item_reviewed(self, fxt_media_prediction_service, fxt_dataset_service):
        project = MagicMock(spec=Project, id=uuid4())
        image = MagicMock(spec=Image, id=uuid4(), type=MediaType.IMAGE)
        dataset_item = MagicMock(spec=DatasetItem, id=uuid4(), user_reviewed=True)
        prediction = MagicMock(spec=BatchInferencePrediction, prediction=[MagicMock(spec=DatasetItemAnnotation)])

        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        fxt_media_prediction_service._create_or_update_dataset_item(
            project=project,
            media=image,
            prediction=prediction,
            model_id=uuid4(),
        )

        fxt_dataset_service.set_dataset_item_annotations.assert_not_called()

    def test_create_dataset_items(self, fxt_media_prediction_service, fxt_media_service, fxt_dataset_service):
        model_id = uuid4()
        image_id = uuid4()
        video_id = uuid4()
        task = MagicMock(spec=Task)
        project = MagicMock(spec=Project, id=uuid4(), task=task)
        image = MagicMock(spec=Image, id=image_id, type=MediaType.IMAGE)
        video = MagicMock(spec=Video, id=video_id, type=MediaType.VIDEO)
        video_frame = MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, video_id=video_id, frame_index=0)
        new_video_frame = MagicMock(spec=VideoFrame, type=MediaType.VIDEO_FRAME, video_id=video_id, frame_index=1)

        image_prediction = BatchInferencePrediction(
            media=BatchInferenceMedia(id=image_id), prediction=[MagicMock(spec=DatasetItemAnnotation)]
        )
        annotated_video_frame_prediction = BatchInferencePrediction(
            media=BatchInferenceMedia(id=video_id, frame_index=0), prediction=[MagicMock(spec=DatasetItemAnnotation)]
        )
        not_annotated_video_frame_image_annotation = MagicMock(spec=DatasetItemAnnotation)

        not_annotated_video_frame = NotAnnotatedVideoFrame(video=video, frame_index=1)
        not_annotated_video_frame_numpy = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)

        loaded_media = LoadedMedia(
            single_media=[image],
            video_frames=[video_frame, not_annotated_video_frame],
        )
        inputs = [
            BatchInferenceInput(media_id=image_id, data=np.random.rand(100, 100, 3)),
            BatchInferenceInput(media_id=video_id, data=np.random.rand(100, 100, 3), frame_index=0),
            BatchInferenceInput(media_id=video_id, data=not_annotated_video_frame_numpy, frame_index=1),
        ]
        batch_inference_result = BatchInferenceResult(
            predictions=[
                image_prediction,
                annotated_video_frame_prediction,
                BatchInferencePrediction(
                    media=BatchInferenceMedia(id=video_id, frame_index=1),
                    prediction=[not_annotated_video_frame_image_annotation],
                ),
            ]
        )

        fxt_media_service.save_video_frame.return_value = new_video_frame

        with patch.object(
            fxt_media_prediction_service, "_create_or_update_dataset_item"
        ) as mock_create_or_update_dataset_item:
            fxt_media_prediction_service._create_dataset_items(
                project=project,
                loaded_media=loaded_media,
                inputs=inputs,
                batch_inference_result=batch_inference_result,
                model_id=model_id,
            )

        # Assert that dataset items for already annotated media (images or video frames) are updated
        mock_create_or_update_dataset_item.assert_has_calls(
            [
                call(project=project, media=image, prediction=image_prediction, model_id=model_id),
                call(
                    project=project,
                    media=video_frame,
                    prediction=annotated_video_frame_prediction,
                    model_id=model_id,
                ),
            ]
        )

        # Assert that new frame is created together with new dataset item
        fxt_media_service.save_video_frame.assert_called_once_with(
            project=project, video=video, frame_index=1, frame_image=ANY
        )
        saved_frame_image = fxt_media_service.save_video_frame.call_args.kwargs["frame_image"]
        assert isinstance(saved_frame_image, PILImage.Image)
        np.testing.assert_array_equal(np.asarray(saved_frame_image), not_annotated_video_frame_numpy)
        fxt_dataset_service.create_dataset_item.assert_called_once_with(
            project_id=project.id,
            task=task,
            media=new_video_frame,
            annotations=[not_annotated_video_frame_image_annotation],
            user_reviewed=False,
            prediction_model_id=model_id,
        )

    @pytest.mark.parametrize("save_predictions", [True, False])
    def test_predict_media(
        self, fxt_media_prediction_service, fxt_label_service, fxt_inference_server, save_predictions
    ):
        model_id = uuid4()
        task = MagicMock(spec=Task)
        project = MagicMock(spec=Project, id=uuid4(), task=task)

        loaded_media = LoadedMedia(single_media=[], video_frames=[])
        inputs = MagicMock(spec=list)
        labels = MagicMock(spec=list)
        batch_inference_result = MagicMock(spec=BatchInferenceResult)

        fxt_label_service.list_all.return_value = labels
        fxt_inference_server.infer_batch.return_value = batch_inference_result

        request = MediaListPredictionRequest(
            model_id=model_id, media=[], save_predictions=save_predictions, device="AUTO"
        )
        device = DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        with (
            patch.object(fxt_media_prediction_service, "_load_media", return_value=loaded_media) as mock_load_media,
            patch.object(
                fxt_media_prediction_service, "_convert_to_inference_input", return_value=inputs
            ) as mock_convert_to_inference_input,
            patch.object(fxt_media_prediction_service, "_create_dataset_items") as mock_create_dataset_items,
        ):
            result = fxt_media_prediction_service.predict_media(project=project, request=request, device=device)

        mock_load_media.assert_called_once_with(project=project, request=request)
        mock_convert_to_inference_input.assert_called_once_with(project=project, loaded_media=loaded_media)
        fxt_label_service.list_all.assert_called_once_with(project_id=project.id)
        fxt_inference_server.set_inference_model.assert_called_once_with(
            project_id=project.id, model_id=model_id, device=device, ttl=10
        )
        fxt_inference_server.infer_batch.assert_called_once_with(labels=labels, inputs=inputs)
        if save_predictions:
            mock_create_dataset_items.assert_called_once_with(
                project=project,
                loaded_media=loaded_media,
                inputs=inputs,
                batch_inference_result=batch_inference_result,
                model_id=request.model_id,
            )
        else:
            mock_create_dataset_items.assert_not_called()
        assert result == batch_inference_result
