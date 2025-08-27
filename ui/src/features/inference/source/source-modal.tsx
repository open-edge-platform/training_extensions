// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, Dialog, DialogTrigger, Heading, Text } from '@geti/ui';

import { ReactComponent as Camera } from '../../../assets/icons/camera.svg';
import { ConnectionPreview, Source } from './source';

export const SourceModal = () => {
    return (
        <DialogTrigger>
            <Button width={'size-2000'} variant={'secondary'}>
                <Text>Input source</Text>
                <Camera fill='white' />
            </Button>
            {(_close) => (
                <Dialog>
                    <Heading>
                        <ConnectionPreview />
                    </Heading>
                    <Content marginTop={'size-200'}>
                        <Source />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
