import { createNetworkFixture, type NetworkFixture } from '@msw/playwright';
import { fromOpenApi } from '@mswjs/source/open-api';
import { test as base } from '@playwright/test';
import { http, HttpResponse } from 'msw';

import spec from '../src/api/openapi-spec.json' with { type: 'json' };

const handlers = await fromOpenApi(JSON.stringify(spec));

interface Fixtures {
    network: NetworkFixture;
}

const defaultHandlers = [
    ...handlers,
    http.get('**/api/models', () => {
        return HttpResponse.json({ models: [{ id: 1, name: 'Test Model' }] });
    }),
    http.post('**/webrtc/offer', () => {
        return HttpResponse.json({
            sdp: 'test',
            type: 'answer',
        });
    }),
    http.post('**/webrtc/input_hook', () => {
        return HttpResponse.json({});
    }),
];

export const test = base.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: defaultHandlers,
    }),
});

export { expect } from '@playwright/test';
