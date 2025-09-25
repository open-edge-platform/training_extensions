// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Item, Picker, TextField } from '@geti/ui';

import { isOnlyDigits, isValidIp } from './utils';

const getIpAddressError = (value: string) => {
    return value === '' || isValidIp(value) ? null : 'Enter a valid IP address';
};

const getPortError = (value: string) => {
    return value === '' || isOnlyDigits(value) ? null : 'Enter a valid port number';
};

export const IpCamera = () => {
    return (
        <Flex direction='column' gap='size-200'>
            <Flex direction='row' gap='size-200'>
                <TextField flex='1' label='IP Address' name='ip_address' validate={getIpAddressError} />
                <TextField flex='1' label='Port' name='port' validate={getPortError} />
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
