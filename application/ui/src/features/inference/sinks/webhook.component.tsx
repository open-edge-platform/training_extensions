// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { Button, Flex, Item, NumberField, Picker, TextField } from '@geti/ui';
import { isEmpty, omitBy } from 'lodash-es';

import { JsonBuilder } from './json-builder.component';
import { OutputFormats } from './output-formats.component';
import { OutputFormat, SinkType, WebhookHttpMethod } from './utils';

type WebhookFormData = {
    name: string;
    timeout: number;
    sink_type: SinkType;
    rate_limit: number;
    webhook_url: string;
    http_method: WebhookHttpMethod;
    output_formats: OutputFormat[];
    headers: Record<string, string>;
};

const getJsonHeaders = (keys: FormDataEntryValue[], values: FormDataEntryValue[]) => {
    const entries = keys.map((key, index) => [key, values[index]]);
    const newObject = Object.fromEntries(entries);

    return omitBy(newObject, isEmpty);
};

export const Webhook = () => {
    const initData = {
        name: 'webhook test',
        timeout: 5000,
        sink_type: SinkType.WEBHOOK,
        rate_limit: 0.2,
        webhook_url: './123',
        http_method: WebhookHttpMethod.POST,
        output_formats: [OutputFormat.IMAGE_ORIGINAL, OutputFormat.PREDICTIONS],
        headers: { Authorization: 'Bearer token' },
    };

    const [state, submitAction, isPending] = useActionState<WebhookFormData, FormData>(async (_prevState, formData) => {
        //Todo: call create endpoint
        const data = {
            name: formData.get('name'),
            headers: getJsonHeaders(formData.getAll('headers-keys'), formData.getAll('headers-values')),
            timeout: formData.get('timeout'),
            sink_type: SinkType.WEBHOOK,
            rate_limit: formData.get('rate_limit'),
            webhook_url: formData.get('webhook_url'),
            http_method: formData.get('http_method'),
            output_formats: formData.getAll('output_formats'),
        } as unknown as WebhookFormData;

        console.log('data', data);
        console.log('_prevState', _prevState);

        return data;
    }, initData);

    console.log('state', state.headers);

    return (
        <form action={submitAction}>
            <Flex direction='column' gap='size-200'>
                <Flex direction={'row'} gap='size-200'>
                    <TextField flex='1' label='Name' name='name' defaultValue={state?.name} />
                    <NumberField
                        label='Rate Limit'
                        name='rate_limit'
                        minValue={0}
                        step={0.1}
                        defaultValue={state?.rate_limit}
                    />
                </Flex>

                <Flex direction={'row'} gap='size-200'>
                    <Picker name='http_method' flex='1' label='HTTP Method' defaultSelectedKey={state?.http_method}>
                        <Item key={WebhookHttpMethod.POST}>{WebhookHttpMethod.POST}</Item>
                        <Item key={WebhookHttpMethod.PUT}>{WebhookHttpMethod.PUT}</Item>
                    </Picker>
                    <NumberField label='Timeout' name='timeout' minValue={0} step={1} defaultValue={state?.timeout} />
                </Flex>

                <TextField width={'100%'} label='Webhook URL' name='webhook_url' defaultValue={state?.webhook_url} />

                <OutputFormats />

                <JsonBuilder title='Headers' keysName='headers-keys' valuesName='headers-values' />

                <Button maxWidth={'size-1000'} type='submit' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </form>
    );
};
