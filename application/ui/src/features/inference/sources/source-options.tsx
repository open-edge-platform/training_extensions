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
import { getImageFolderData } from './util';
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
                { label: 'Webcam', value: 'webcam', content: <Webcam />, icon: <WebcamIcon width={'24px'} /> },
                {
                    label: 'IP Camera',
                    value: 'ip_camera',
                    content: <IpCamera />,
                    icon: <IpCameraIcon width={'24px'} />,
                },
                { label: 'GenICam', value: 'gen_i_cam', content: <ImageFolder />, icon: <GenICam width={'24px'} /> },
                { label: 'Video file', value: 'video_file', content: <ImageFolder />, icon: <Video width={'24px'} /> },
                {
                    label: 'Images folder',
                    value: 'images_folder',
                    content: <ImageFolder config={getImageFolderData(sources)} />,
                    icon: <Image width={'24px'} />,
                },
            ]}
        />
    );
};
