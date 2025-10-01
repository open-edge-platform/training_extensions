// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { Button, Flex, Form, NumberField, Switch, TextField } from '@geti/ui';

import { OutputFormats } from '../output-formats.component';
import { OutputFormat, SinkType } from '../utils';

type MqttFormData = {
    name: string;
    topic: string;
    timeout: number;
    sink_type: SinkType;
    rate_limit: number;
    broker_host: string;
    output_formats: OutputFormat[];
    broker_port: number;
    auth_required: boolean;
};

export const Mqtt = () => {
    const initData = {
        name: 'mqtt test',
        topic: 'test/topic',
        timeout: 5000,
        sink_type: SinkType.MQTT,
        rate_limit: 0.2,
        broker_host: 'localhost',
        broker_port: 1883,
        auth_required: false,
        output_formats: [OutputFormat.IMAGE_ORIGINAL, OutputFormat.PREDICTIONS],
    };

    const [state, submitAction, isPending] = useActionState<MqttFormData, FormData>(async (_prevState, formData) => {
        //Todo: call create endpoint
        const data = {
            name: formData.get('name'),
            output_formats: formData.getAll('output_formats'),
            topic: formData.get('topic'),
            timeout: formData.get('timeout'),
            sink_type: SinkType.MQTT,
            rate_limit: formData.get('rate_limit'),
            broker_host: formData.get('broker_host'),
            broker_port: formData.get('broker_port'),
            auth_required: formData.get('auth_required') === 'on',
        } as unknown as MqttFormData;

        return data;
    }, initData);

    return (
        <Form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <TextField width={'100%'} label='Name' name='name' defaultValue={state?.name} />

                <TextField width={'100%'} label='Broker Host' name='broker_host' defaultValue={state?.broker_host} />

                <Flex direction={'row'} gap='size-200'>
                    <NumberField
                        label='Broker Port'
                        name='broker_port'
                        minValue={0}
                        step={1}
                        defaultValue={state?.broker_port}
                    />

                    <NumberField
                        label='Rate Limit'
                        name='rate_limit'
                        minValue={0}
                        step={0.1}
                        defaultValue={state?.rate_limit}
                    />

                    <NumberField label='Timeout' name='timeout' minValue={0} step={1} defaultValue={state?.timeout} />
                </Flex>

                <Flex direction={'row'} gap='size-200'>
                    <TextField flex='1' label='Topic' name='topic' defaultValue={state?.topic} />
                    <Switch name='auth_required' defaultSelected={state?.auth_required} alignSelf={'end'}>
                        Auth Required
                    </Switch>
                </Flex>

                <OutputFormats />

                <Button maxWidth={'size-1000'} type='submit' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
