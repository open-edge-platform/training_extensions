// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Form, Item, Picker, TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { isOnlyDigits, WebcamSourceConfig } from '../util';

type WebcamProps = {
    config?: WebcamSourceConfig;
};

const initConfig: WebcamSourceConfig = {
    id: '',
    name: '',
    source_type: 'webcam',
    device_id: 0,
};

const LIST_OF_CODECS = ['XVID', 'MJPG', 'YUY2', 'H264', 'H265', 'MP4V', 'DIVX'];

export const Webcam = ({ config = initConfig }: WebcamProps) => {
    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: isEmpty(config?.id),
        bodyFormatter: (formData: FormData) => {
            const codec = formData.get('codec');
            const body: WebcamSourceConfig = {
                id: String(formData.get('id')),
                name: String(formData.get('name')),
                source_type: 'webcam',
                device_id: Number(formData.get('device_id')),
                ...(codec ? { codec: String(codec) } : {}),
            };

            return body;
        },
    });

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={state?.id} />
                <TextField width={'100%'} label='Name' name='name' defaultValue={state?.name} />

                <TextField
                    width='100%'
                    label='Webcam device id'
                    name='device_id'
                    defaultValue={String(state?.device_id)}
                    validate={(value) => (isOnlyDigits(value) ? '' : 'Only digits are allowed')}
                />

                <Picker
                    placeholder={'Select codec'}
                    name='codec'
                    label='Codec'
                    aria-label={'List of codecs'}
                    defaultSelectedKey={String(state?.codec)}
                >
                    {LIST_OF_CODECS.map((codec) => {
                        return <Item key={codec}>{codec}</Item>;
                    })}
                </Picker>

                <Button type='submit' maxWidth='size-1000' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
