// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';
import { ReactComponent as GenICam } from '../../../assets/icons/genicam.svg';
import { ReactComponent as Image } from '../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCameraIcon } from '../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../assets/icons/webcam.svg';
import { DisclosureGroup } from './disclosure-group.component';
import { ImageFolder } from './image-folder/image-folder.component';
import { IpCamera } from './ip-camera/ip-camera.component';
import { getImageFolderData, getIpCameraData, getVideoFileData, getWebcamData } from './util';
import { VideoFile } from './video-file/video-file.component';
import { Webcam } from './webcam/webcam.component';

export const SourceOptions = () => {
    const projectId = useProjectIdentifier();

    const sourcesQuery = $api.useSuspenseQuery('get', '/api/sources');
    const pipeline = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });
    const sources = sourcesQuery.data ?? [];

    return (
        <DisclosureGroup
            defaultActiveInput={pipeline.data?.source?.source_type ?? null}
            items={[
                {
                    label: 'Webcam',
                    value: 'webcam',
                    icon: <WebcamIcon width={'24px'} />,
                    content: <Webcam config={getWebcamData(sources)} />,
                },
                {
                    label: 'IP Camera',
                    value: 'ip_camera',
                    icon: <IpCameraIcon width={'24px'} />,
                    content: <IpCamera config={getIpCameraData(sources)} />,
                },
                { label: 'GenICam', value: 'gen_i_cam', content: <ImageFolder />, icon: <GenICam width={'24px'} /> },
                {
                    label: 'Video file',
                    value: 'video_file',
                    content: <VideoFile config={getVideoFileData(sources)} />,
                    icon: <Video width={'24px'} />,
                },
                {
                    label: 'Images folder',
                    value: 'images_folder',
                    icon: <Image width={'24px'} />,
                    content: <ImageFolder config={getImageFolderData(sources)} />,
                },
            ]}
        />
    );
};
