// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, Content, Dialog, DialogTrigger, Text } from '@geti/ui';

import { ReactComponent as Camera } from '../../../assets/icons/camera.svg';
import { ReactComponent as GenICam } from '../../../assets/icons/genicam.svg';
import { ReactComponent as Image } from '../../../assets/icons/images-folder.svg';
import { ReactComponent as IpCameraIcon } from '../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../assets/icons/webcam.svg';
import { DisclosureGroup } from './disclosure-group.component';
import { ImageFolder } from './image-folder/image-folder.component';
import { IpCamera } from './ip-camera/ip-camera.component';
import { Webcam } from './webcam/webcam.component';

const inputs = [
    { label: 'Webcam', value: 'webcam', content: <Webcam />, icon: <WebcamIcon width={'24px'} /> },
    { label: 'IP Camera', value: 'ip-camera', content: <IpCamera />, icon: <IpCameraIcon width={'24px'} /> },
    { label: 'GenICam', value: 'gen-i-cam', content: <ImageFolder />, icon: <GenICam width={'24px'} /> },
    { label: 'Video file', value: 'video-file', content: <ImageFolder />, icon: <Video width={'24px'} /> },
    {
        label: 'Image folder',
        value: 'image-folder',
        content: <ImageFolder />,
        icon: <Image width={'24px'} />,
    },
];

export const SourceModal = () => {
    const [activeInput, setActiveInput] = useState<string | null>(null);

    const handleActiveInputChange = (value: string) => {
        setActiveInput((prevValue) => (value !== prevValue ? value : null));
    };

    return (
        <DialogTrigger type='popover'>
            <Button width={'size-2000'} variant={'secondary'}>
                <Camera fill='white' />
                <Text width={'auto'} marginStart={'size-100'}>
                    Input source
                </Text>
            </Button>
            <Dialog>
                <Content>
                    <DisclosureGroup items={inputs} value={activeInput} onChange={handleActiveInputChange} />
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
