// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Item, NumberField, Picker, TextField } from '@geti/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitFields } from '../rate-limit/rate-limit-fields.component';
import { WebhookHttpMethod, WebhookSinkConfig } from '../utils';
import { HeaderKeyValueBuilder } from './header-key-value-builder.component';

type WebhookProps = {
    defaultState: WebhookSinkConfig;
};

export const Webhook = ({ defaultState }: WebhookProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <Flex direction={'row'} gap='size-200'>
                <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
                <TextField flex='1' label='Name' name='name' defaultValue={defaultState.name || 'Webhook sink'} />
            </Flex>
            <Flex>
                <RateLimitFields rateLimit={defaultState.rate_limit} />
            </Flex>

            <Flex direction={'row'} gap='size-200'>
                <Picker name='http_method' flex='1' label='HTTP Method' defaultSelectedKey={defaultState.http_method}>
                    <Item key={WebhookHttpMethod.POST}>{WebhookHttpMethod.POST}</Item>
                    <Item key={WebhookHttpMethod.PATCH}>{WebhookHttpMethod.PATCH}</Item>
                    <Item key={WebhookHttpMethod.PUT}>{WebhookHttpMethod.PUT}</Item>
                </Picker>
                <NumberField label='Timeout' name='timeout' minValue={1} step={1} defaultValue={defaultState.timeout} />
            </Flex>

            <TextField
                isRequired
                width={'100%'}
                label='Webhook URL'
                name='webhook_url'
                defaultValue={defaultState.webhook_url}
            />

            <OutputFormats config={defaultState.output_formats} />

            <HeaderKeyValueBuilder
                config={defaultState.headers ?? undefined}
                title='Headers'
                keysName='headers-keys'
                valuesName='headers-values'
                key={JSON.stringify(defaultState.headers) + defaultState.id}
            />
        </Flex>
    );
};
