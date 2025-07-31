import '@testing-library/jest-dom';

import { server } from './msw-node-setup';

beforeAll(() => {
    server.listen();
});

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => server.close());
