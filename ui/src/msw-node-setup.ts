import { setupServer } from 'msw/node';

import { handlers, http } from './api/utils';

// Initialize msw's mock server with the handlers
const server = setupServer(...handlers);

// For use in test setup/teardown
export { server, http };
