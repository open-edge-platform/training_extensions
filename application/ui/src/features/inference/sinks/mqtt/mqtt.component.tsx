// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Form, NumberField, Switch, TextField } from '@geti/ui';
import { isEmpty } from 'lodash';

import { useSinkAction } from '../hooks/use-sink-action.hook';
import { OutputFormats } from '../output-formats.component';
import { MqttSinkConfig, SinkOutputFormats } from '../utils';

type MqttProps = {
    config?: MqttSinkConfig;
};
const initConfig: MqttSinkConfig = {
    id: 'mqtt-id',
    name: '',
    topic: '',
    sink_type: 'mqtt',
    rate_limit: 0,
    broker_host: '',
    broker_port: 0,
    auth_required: false,
    output_formats: [],
};

export const Mqtt = ({ config = initConfig }: MqttProps) => {
    const [state, submitAction, isPending] = useSinkAction<MqttSinkConfig>({
        config,
        isNewSink: isEmpty(config?.id),
        bodyFormatter: (formData: FormData): MqttSinkConfig => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            topic: String(formData.get('topic')),
            sink_type: 'mqtt',
            rate_limit: formData.get('rate_limit') ? Number(formData.get('rate_limit')) : 0,
            broker_host: String(formData.get('broker_host')),
            broker_port: formData.get('broker_port') ? Number(formData.get('broker_port')) : 0,
            auth_required: formData.get('auth_required') === 'on' ? true : false,
            output_formats: formData.getAll('output_formats') as SinkOutputFormats,
        }),
    });

    return (
        <Form action={submitAction}>
            <TextField isHidden label='id' name='id' defaultValue={state?.id} />

            <Flex direction='column' gap='size-200'>
                <TextField width={'100%'} label='Name' name='name' defaultValue={state?.name} />

                <TextField width={'100%'} label='Broker Host' name='broker_host' defaultValue={state?.broker_host} />

                <Flex direction={'row'} gap='size-200'>
                    <TextField flex='1' label='Topic' name='topic' defaultValue={state?.topic} />
                    <NumberField
                        label='Broker Port'
                        name='broker_port'
                        minValue={0}
                        step={1}
                        defaultValue={state?.broker_port}
                    />
                </Flex>

                <Flex direction={'row'} gap='size-200' justifyContent='space-between'>
                    <NumberField
                        label='Rate Limit'
                        name='rate_limit'
                        minValue={0}
                        step={0.1}
                        defaultValue={state?.rate_limit ?? undefined}
                    />
                    <Switch
                        name='auth_required'
                        alignSelf='end'
                        aria-label='Require Authentication'
                        defaultSelected={state?.auth_required}
                        key={state?.auth_required ? 'true' : 'false'}
                    >
                        Auth Required
                    </Switch>
                </Flex>

                <OutputFormats config={state?.output_formats} />

                <Button maxWidth={'size-1000'} type='submit' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
