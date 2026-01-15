// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ReactComponent as GenICam } from '../../../assets/icons/genicam.svg';
import { ReactComponent as Image } from '../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCameraIcon } from '../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../assets/icons/webcam.svg';
import { AddSource } from './add-source/add-source.component';
import { DisclosureGroup } from './disclosure-group.component';
import { ImageFolder } from './image-folder/image-folder.component';
import { getImagesFolderInitialConfig, imagesFolderBodyFormatter } from './image-folder/utils';
import { IpCamera } from './ip-camera/ip-camera.component';
import { getIpCameraInitialConfig, ipCameraBodyFormatter } from './ip-camera/utils';
import { ImagesFolderSourceConfig, IPCameraSourceConfig, USBCameraSourceConfig, VideoFileSourceConfig } from './util';
import { getVideoFileInitialConfig, videoFileBodyFormatter } from './video-file/utils';
import { VideoFile } from './video-file/video-file.component';
import { getUsbCameraInitialConfig, usbCameraBodyFormatter } from './webcam/utils';
import { WebcamFields } from './webcam/webcam-fields.component';

interface SourceOptionsProps {
    onSaved: () => void;
    hasHeader: boolean;
    children: ReactNode;
}

export const SourceOptions = ({ onSaved, hasHeader, children }: SourceOptionsProps) => {
    return (
        <>
            {hasHeader && children}

            <DisclosureGroup
                defaultActiveInput={null}
                items={[
                    {
                        label: 'Webcam',
                        value: 'usb_camera',
                        icon: <WebcamIcon width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getUsbCameraInitialConfig()}
                                componentFields={(state: USBCameraSourceConfig) => (
                                    <WebcamFields defaultState={state} />
                                )}
                                bodyFormatter={usbCameraBodyFormatter}
                            />
                        ),
                    },
                    {
                        label: 'IP Camera',
                        value: 'ip_camera',
                        icon: <IpCameraIcon width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getIpCameraInitialConfig()}
                                componentFields={(state: IPCameraSourceConfig) => <IpCamera defaultState={state} />}
                                bodyFormatter={ipCameraBodyFormatter}
                            />
                        ),
                    },
                    {
                        label: 'GenICam',
                        value: 'gen_i_cam',

                        icon: <GenICam width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getImagesFolderInitialConfig()}
                                componentFields={(state: ImagesFolderSourceConfig) => (
                                    <ImageFolder defaultState={state} />
                                )}
                                bodyFormatter={imagesFolderBodyFormatter}
                            />
                        ),
                    },
                    {
                        label: 'Video file',
                        value: 'video_file',
                        icon: <Video width={'24px'} />,

                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getVideoFileInitialConfig()}
                                componentFields={(state: VideoFileSourceConfig) => <VideoFile defaultState={state} />}
                                bodyFormatter={videoFileBodyFormatter}
                            />
                        ),
                    },
                    {
                        label: 'Images folder',
                        value: 'images_folder',
                        icon: <Image width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getImagesFolderInitialConfig()}
                                componentFields={(state: ImagesFolderSourceConfig) => (
                                    <ImageFolder defaultState={state} />
                                )}
                                bodyFormatter={imagesFolderBodyFormatter}
                            />
                        ),
                    },
                ]}
            />
        </>
    );
};
