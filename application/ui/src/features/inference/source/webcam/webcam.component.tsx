// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, Flex, TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

export const Webcam = () => {
    const [deviceId, setDeviceId] = useState('');

    return (
        <Flex direction='column' gap='size-200'>
            <TextField label='Webcam device id' name='device_id' value={deviceId} onChange={setDeviceId} />
            <Button maxWidth={'size-1000'} isDisabled={isEmpty(deviceId)}>
                Apply
            </Button>
        </Flex>
    );
};
