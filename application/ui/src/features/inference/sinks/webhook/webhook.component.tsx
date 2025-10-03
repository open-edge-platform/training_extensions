// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Form, Item, NumberField, Picker, TextField } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useSinkAction } from '../hooks/use-sink-action.hook';
import { KeyValueBuilder } from '../key-value-builder.component';
import { OutputFormats } from '../output-formats.component';
import { getObjectFromFormData, SinkOutputFormats, WebhookHttpMethod, WebhookSinkConfig } from '../utils';

type WebhookProps = {
    config?: WebhookSinkConfig;
};

const initConfig: WebhookSinkConfig = {
    name: '',
    timeout: 0,
    sink_type: 'webhook',
    rate_limit: 0,
    webhook_url: '',
    http_method: WebhookHttpMethod.POST,
    output_formats: [],
    headers: {},
};
export const Webhook = ({ config = initConfig }: WebhookProps) => {
    const [state, submitAction, isPending] = useSinkAction<WebhookSinkConfig>({
        config,
        isNewSink: isEmpty(config?.id),
        bodyFormatter: (formData: FormData): WebhookSinkConfig => ({
            id: String(formData.get('id')),
            name: String(formData.get('name')),
            headers: getObjectFromFormData(formData.getAll('headers-keys'), formData.getAll('headers-values')),
            timeout: Number(formData.get('timeout')),
            sink_type: 'webhook',
            rate_limit: Number(formData.get('rate_limit')),
            webhook_url: String(formData.get('webhook_url')),
            http_method: formData.get('http_method') as WebhookHttpMethod,
            output_formats: formData.getAll('output_formats') as SinkOutputFormats,
        }),
    });

    return (
        <Form action={submitAction}>
            <TextField isHidden label='id' name='id' defaultValue={state?.id} />

            <Flex direction='column' gap='size-200'>
                <Flex direction={'row'} gap='size-200'>
                    <TextField flex='1' label='Name' name='name' defaultValue={state?.name} />
                    <NumberField
                        label='Rate Limit'
                        name='rate_limit'
                        minValue={0}
                        step={0.1}
                        defaultValue={state?.rate_limit ?? undefined}
                    />
                </Flex>

                <Flex direction={'row'} gap='size-200'>
                    <Picker name='http_method' flex='1' label='HTTP Method' defaultSelectedKey={state?.http_method}>
                        <Item key={WebhookHttpMethod.POST}>{WebhookHttpMethod.POST}</Item>
                        <Item key={WebhookHttpMethod.PATCH}>{WebhookHttpMethod.PATCH}</Item>
                        <Item key={WebhookHttpMethod.PUT}>{WebhookHttpMethod.PUT}</Item>
                    </Picker>
                    <NumberField label='Timeout' name='timeout' minValue={0} step={1} defaultValue={state?.timeout} />
                </Flex>

                <TextField width={'100%'} label='Webhook URL' name='webhook_url' defaultValue={state?.webhook_url} />

                <OutputFormats config={state?.output_formats} />

                <KeyValueBuilder
                    config={state?.headers ?? undefined}
                    title='Headers'
                    keysName='headers-keys'
                    valuesName='headers-values'
                    key={JSON.stringify(state?.headers) + state?.id}
                />

                <Button maxWidth={'size-1000'} type='submit' isDisabled={isPending}>
                    Apply
                </Button>
            </Flex>
        </Form>
    );
};
