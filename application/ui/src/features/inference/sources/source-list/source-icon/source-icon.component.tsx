// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactComponent as GenICamIcon } from '../../../../../assets/icons/genicam.svg';
import { ReactComponent as ImageFolderIcon } from '../../../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCameraIcon } from '../../../../../assets/icons/ip-camera.svg';
import { ReactComponent as VideoFileIcon } from '../../../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../../../assets/icons/webcam.svg';

interface SourceIconProps {
    type: string;
}

export const SourceIcon = ({ type }: SourceIconProps) => {
    if (type === 'usb_camera') {
        return <WebcamIcon />;
    }

    if (type === 'ip_camera') {
        return <IpCameraIcon />;
    }

    if (type === 'video_file') {
        return <VideoFileIcon />;
    }

    if (type === 'gen_i_cam') {
        return <GenICamIcon />;
    }

    return <ImageFolderIcon />;
};
