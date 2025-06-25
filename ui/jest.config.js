import sharedConfig from '@geti/config/test';

export default {
    projects: [
        {
            ...sharedConfig,
            displayName: '@geti/edge-ui',
            roots: ['<rootDir>/src'],
            testMatch: ['<rootDir>/src/**/*.{test,spec}.{js,jsx,ts,tsx}'],
            setupFilesAfterEnv: ['<rootDir>/src/setup-tests.tsx'],
        },
    ],
    collectCoverageFrom: ['<rootDir>/src/**/*{test,spec}.{js,jsx,ts,tsx}'],
    coverageReporters: ['clover', 'json', 'json-summary'],
    coverageThreshold: {
        global: {
            lines: 75,
        },
    },
};
