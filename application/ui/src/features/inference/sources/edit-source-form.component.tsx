// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { EditSource } from './edit-source/edit-source.component';
import { ImageFolder } from './image-folder/image-folder.component';
import { imagesFolderBodyFormatter } from './image-folder/utils';
import { IpCamera } from './ip-camera/ip-camera.component';
import { ipCameraBodyFormatter } from './ip-camera/utils';
import { ImagesFolderSourceConfig, SourceConfig, VideoFileSourceConfig } from './util';
import { videoFileBodyFormatter } from './video-file/utils';
import { VideoFile } from './video-file/video-file.component';
import { usbCameraBodyFormatter } from './webcam/utils';
import { WebcamFields } from './webcam/webcam-fields.component';

interface EditSourceFormProps {
    config: SourceConfig;
    onSaved: () => void;
    onBackToList: () => void;
}

export const EditSourceForm = ({ config, onSaved, onBackToList }: EditSourceFormProps) => {
    if (config.source_type === 'usb_camera') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <WebcamFields defaultState={state} />}
                bodyFormatter={usbCameraBodyFormatter}
            />
        );
    }

    if (config.source_type === 'ip_camera') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <IpCamera defaultState={state} />}
                bodyFormatter={ipCameraBodyFormatter}
            />
        );
    }

    if (config.source_type === 'video_file') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state: VideoFileSourceConfig) => <VideoFile defaultState={state} />}
                bodyFormatter={videoFileBodyFormatter}
            />
        );
    }

    return (
        <EditSource
            onSaved={onSaved}
            onBackToList={onBackToList}
            config={config as ImagesFolderSourceConfig}
            componentFields={(state: ImagesFolderSourceConfig) => <ImageFolder defaultState={state} />}
            bodyFormatter={imagesFolderBodyFormatter}
        />
    );
};
