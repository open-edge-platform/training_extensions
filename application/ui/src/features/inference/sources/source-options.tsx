// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ReactComponent as IpCameraIcon } from '../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../assets/icons/webcam.svg';
import { IPCameraSourceConfig, USBCameraSourceConfig, VideoFileSourceConfig } from '../../../constants/shared-types';
import { AddSource } from './add-source/add-source.component';
import { DisclosureGroup } from './disclosure-group.component';
import { IpCamera } from './ip-camera/ip-camera.component';
import { getIpCameraInitialConfig, ipCameraBodyFormatter } from './ip-camera/utils';
import { UsbCamera } from './usb-camera/usb-camera-fields.component';
import { getUsbCameraInitialConfig, usbCameraBodyFormatter } from './usb-camera/utils';
import { getVideoFileInitialConfig, videoFileBodyFormatter } from './video-file/utils';
import { VideoFile } from './video-file/video-file.component';

interface SourceOptionsProps {
    onSaved: () => void;
    hasHeader: boolean;
    children: ReactNode;
    existingNames?: string[];
}

export const SourceOptions = ({ onSaved, hasHeader, children, existingNames = [] }: SourceOptionsProps) => {
    return (
        <>
            {hasHeader && children}

            <DisclosureGroup
                defaultActiveInput={null}
                items={[
                    {
                        label: 'USB Camera',
                        value: 'usb_camera',
                        icon: <WebcamIcon width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getUsbCameraInitialConfig(existingNames)}
                                componentFields={(state: USBCameraSourceConfig) => <UsbCamera defaultState={state} />}
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
                                config={getIpCameraInitialConfig(existingNames)}
                                componentFields={(state: IPCameraSourceConfig) => <IpCamera defaultState={state} />}
                                bodyFormatter={ipCameraBodyFormatter}
                            />
                        ),
                    },
                    // TODO: Reenable after MVP
                    // {
                    //     label: 'GenICam',
                    //     value: 'gen_i_cam',

                    //     icon: <GenICam width={'24px'} />,
                    //     content: (
                    //         <AddSource
                    //             onSaved={onSaved}
                    //             config={getImagesFolderInitialConfig()}
                    //             componentFields={(state: ImagesFolderSourceConfig) => (
                    //                 <ImageFolder defaultState={state} />
                    //             )}
                    //             bodyFormatter={imagesFolderBodyFormatter}
                    //         />
                    //     ),
                    // },
                    {
                        label: 'Video file',
                        value: 'video_file',
                        icon: <Video width={'24px'} />,

                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getVideoFileInitialConfig(existingNames)}
                                componentFields={(state: VideoFileSourceConfig) => <VideoFile defaultState={state} />}
                                bodyFormatter={videoFileBodyFormatter}
                            />
                        ),
                    },
                    // TODO: Reenable after MVP
                    // {
                    //     label: 'Images folder',
                    //     value: 'images_folder',
                    //     icon: <Image width={'24px'} />,
                    //     content: (
                    //         <AddSource
                    //             onSaved={onSaved}
                    //             config={getImagesFolderInitialConfig()}
                    //             componentFields={(state: ImagesFolderSourceConfig) => (
                    //                 <ImageFolder defaultState={state} />
                    //             )}
                    //             bodyFormatter={imagesFolderBodyFormatter}
                    //         />
                    //     ),
                    // },
                ]}
            />
        </>
    );
};
