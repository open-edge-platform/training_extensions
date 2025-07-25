import { fromOpenApi } from '@mswjs/source/open-api';
import { setupServer } from 'msw/node';

import spec from './api/openapi-spec.json' with { type: 'json' };

const handlers = await fromOpenApi(JSON.stringify(spec));

// Initialize msw's mock server with the handlers
const server = setupServer(...handlers);

// For use in test setup/teardown
export { server };
