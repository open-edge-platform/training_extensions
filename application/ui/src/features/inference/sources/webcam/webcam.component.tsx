// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, useRef, useState } from 'react';

import { ActionButton, Button, Flex, Form, Item, Loading, Picker, TextField } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { $api } from '../../../../api/client';
import { useSourceAction } from '../hooks/use-source-action.hook';
import { USBCameraSourceConfig } from '../util';

type WebcamProps = {
    config?: USBCameraSourceConfig;
};

const initConfig: USBCameraSourceConfig = {
    id: '',
    name: '',
    source_type: 'usb_camera',
    device_id: 0,
};

export const Webcam = ({ config = initConfig }: WebcamProps) => {
    const [name, setName] = useState(config.name);
    const isSystemName = useRef(isEmpty(config.name));

    const {
        data: cameraDevices,
        isLoading,
        isRefetching,
        refetch,
    } = $api.useQuery('get', '/api/system/devices/camera');

    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: isEmpty(config?.id),
        bodyFormatter: (formData: FormData) => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            source_type: 'usb_camera',
            device_id: Number(formData.get('device_id')),
        }),
    });

    const devices = (cameraDevices ?? []).map((device) => ({
        id: device.index,
        name: device.name,
    }));

    const handleNameChange = (value: string) => {
        setName(value);
        isSystemName.current = false;
    };

    const handleSelectionChange = (key: Key | null) => {
        const device = devices.find(({ id }) => id === Number(key));

        if (device && isSystemName.current) {
            setName(device.name);
        }
    };

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={state?.id} />
                <TextField isHidden label='name' name='name' value={name} />
                <TextField
                    isRequired
                    width='100%'
                    label='Name'
                    name='name_display'
                    value={name}
                    onChange={handleNameChange}
                />

                <Flex alignItems='end' gap='size-200'>
                    <Picker
                        flex='1'
                        isRequired
                        label='Camera'
                        name='device_id'
                        items={devices}
                        isLoading={isLoading}
                        aria-label='Camera list'
                        defaultSelectedKey={String(config.device_id)}
                        onSelectionChange={handleSelectionChange}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>

                    <ActionButton
                        isQuiet
                        onPress={() => refetch()}
                        aria-label='Refresh Cameras'
                        isDisabled={isLoading || isRefetching}
                    >
                        {isRefetching ? <Loading mode={'inline'} size='S' /> : <Refresh />}
                    </ActionButton>
                </Flex>

                <Button type='submit' maxWidth='size-1000' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
