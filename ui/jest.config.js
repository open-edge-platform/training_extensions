import sharedConfig from '@geti/config/test';

export default {
    projects: [
        {
            ...sharedConfig,
            displayName: '@geti/edge-ui',
            roots: ['<rootDir>/src'],
            testMatch: ['<rootDir>/src/**/*.{test,spec}.{js,jsx,ts,tsx}'],
            setupFilesAfterEnv: ['<rootDir>/src/setup-tests.tsx'],
            transform: {
                '^.+/(web-rtc-connection|api/client)\.ts$': [
                    'ts-jest',
                    {
                        diagnostics: {
                            ignoreCodes: [1343],
                        },
                        astTransformers: {
                            before: [
                                {
                                    path: 'ts-jest-mock-import-meta',
                                    options: { metaObjectReplacement: { url: 'some-url', env: 'some-env' } },
                                },
                            ],
                        },
                    },
                ],
                ...sharedConfig.transform,
            },
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
