// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Item, NumberField, Picker, TextField } from '@geti/ui';

import { isValidIp } from './utils';

const getIpAddressError = (value: string) => {
    return value === '' || isValidIp(value) ? null : 'Enter a valid IP address';
};

export const IpCamera = () => {
    return (
        <Flex direction='column' gap='size-200'>
            <Flex direction='row' gap='size-200'>
                <TextField flex='1' label='IP Address' name='ip_address' validate={getIpAddressError} />
                <NumberField name='port' label='Port' minValue={0} step={1} />
            </Flex>

            <TextField width={'100%'} label='Stream Path:' name='stream_path' />

            <Picker width={'100%'} label='Protocol' name='protocol'>
                <Item key='rtsp'>RTSP</Item>
                <Item key='http'>HTTP</Item>
                <Item key='https'>HTTPS</Item>
            </Picker>

            <Button maxWidth={'size-1000'}>Apply</Button>
        </Flex>
    );
};
