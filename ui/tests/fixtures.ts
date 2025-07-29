import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { fromOpenApi } from '@mswjs/source/open-api';
import { expect, test as testBase } from '@playwright/test';
import { createOpenApiHttp, OpenApiHttpHandlers } from 'openapi-msw';

import { paths } from '../src/api/openapi-spec';
import spec from '../src/api/openapi-spec.json' with { type: 'json' };

const handlers = await fromOpenApi(JSON.stringify(spec).replace(/}:/g, '}//:'));

interface Fixtures {
    network: NetworkFixture;
}

const getOpenApiHttp = (): OpenApiHttpHandlers<paths> => {
    const http = createOpenApiHttp<paths>({
        baseUrl: process.env.PUBLIC_API_BASE_URL ?? 'http://localhost:3000',
    });

    return {
        ...http,
        post: (path, ...other) => {
            // @ts-expect-error MSW internal parsing function does not accept paths like
            // `/api/models/{model_name}:activate`
            // to get around this we escape the colon character with `\\`
            // @see https://github.com/mswjs/msw/discussions/739
            return http.post(path.replace('}:', '}\\:'), ...other);
        },
    };
};

const http = getOpenApiHttp();

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: [
            ...handlers,
            http.get('/api/system/metrics/memory', ({ response }) => {
                return response(200).json({});
            }),
            http.get('/api/models', ({ response }) => {
                return response(200).json({
                    active_model: '1',
                    available_models: [],
                });
            }),
            http.post('/api/webrtc/offer', ({ response }) => {
                // Schema is empty, so we return an empty object
                return response(200).json({} as never);
            }),
            http.post('/api/input_hook', ({ response }) => {
                // Schema is empty, so we return an empty object
                return response(200).json({} as never);
            }),
        ],
    }),
});

export { expect, http, test };
