// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FlatCompat } from '@eslint/eslintrc';
import js from '@eslint/js';
import sharedEslintConfig from '@geti/config/lint';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const currentYear = new Date().getFullYear();
const allowedYears = Array.from({ length: Math.max(currentYear - 2025 + 1, 1) }, (_, index) => 2025 + index);
const compat = new FlatCompat({
    baseDirectory: dirname,
    recommendedConfig: js.configs.recommended,
    allConfig: js.configs.all,
});

export default [
    {
        ignores: [...sharedEslintConfig[0].ignores, 'src/api/openapi-spec.d.ts'],
    },
    ...sharedEslintConfig,
    {
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: [
                        {
                            name: '@adobe/react-spectrum',
                            message: 'Use component from the @geti/ui folder instead.',
                        },
                    ],
                    patterns: [
                        {
                            group: ['@react-spectrum'],
                            message: 'Use component from the @geti/ui folder instead.',
                        },
                        {
                            group: ['@react-types/*'],
                            message: 'Use type from the @geti/ui folder instead.',
                        },
                        {
                            group: ['@spectrum-icons'],
                            message: 'Use icons from the @geti/ui/icons folder instead.',
                        },
                        {
                            group: ['src/*'],
                            message: 'Use relative imports instead of absolute "src/" imports.',
                        },
                    ],
                },
            ],
            'header/header': [
                'warn',
                'line',
                [
                    {
                        pattern: ` Copyright \\(C\\) ((?:${allowedYears.join('|')})|2025-(?:${allowedYears.join('|')})) Intel Corporation`,
                        template: ` Copyright (C) 2025-${currentYear} Intel Corporation`,
                    },
                    ' SPDX-License-Identifier: Apache-2.0',
                ],
            ],
        },
    },
    {
        files: ['**/*.test.ts', '**/*.test.tsx', '**/*mock*.ts', '**/*.spec.ts'],
        rules: {
            'max-len': ['off'],
        },
    },
    {
        // Contain Tauri-specific APIs inside `*.tauri.{ts,tsx}` files under
        // `src/platform/`. The bundler picks those files for the Tauri build
        // via `resolve.extensions`, so only they should ever import
        // `@tauri-apps/*`. Everywhere else, consume the capability module
        // (e.g. `import { downloadFile } from '../platform/download-file'`)
        // and let the bundler swap the implementation per target.
        files: ['src/**/*.{ts,tsx}'],
        ignores: ['src/platform/**/*.tauri.{ts,tsx}'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: [
                        {
                            name: '@adobe/react-spectrum',
                            message: 'Use component from the @geti/ui folder instead.',
                        },
                    ],
                    patterns: [
                        {
                            group: ['@react-spectrum'],
                            message: 'Use component from the @geti/ui folder instead.',
                        },
                        {
                            group: ['@react-types/*'],
                            message: 'Use type from the @geti/ui folder instead.',
                        },
                        {
                            group: ['@spectrum-icons'],
                            message: 'Use icons from the @geti/ui/icons folder instead.',
                        },
                        {
                            group: ['src/*'],
                            message: 'Use relative imports instead of absolute "src/" imports.',
                        },
                        {
                            group: ['@tauri-apps/*'],
                            message:
                                'Import Tauri plugins only from `*.tauri.{ts,tsx}` files under src/platform/. ' +
                                'Consumers should import the capability module (e.g. ../platform/download-file) ' +
                                'so the bundler can swap implementations per build target.',
                        },
                    ],
                },
            ],
        },
    },
    ...compat.extends('plugin:playwright/playwright-test').map((config) => ({
        ...config,
        files: ['tests/**/*.ts'],
    })),
    {
        files: ['tests/**/*.ts'],

        rules: {
            'playwright/no-wait-for-selector': ['off'],
            'playwright/no-conditional-expect': ['off'],
            'playwright/no-standalone-expect': ['off'],
            'playwright/missing-playwright-await': ['warn'],
            'playwright/valid-expect': ['warn'],
            'playwright/no-useless-not': ['warn'],
            'playwright/no-page-pause': ['warn'],
            'playwright/prefer-to-have-length': ['warn'],
            'playwright/no-conditional-in-test': ['off'],
            'playwright/expect-expect': ['off'],
            'playwright/no-skipped-test': ['off'],
            'playwright/no-wait-for-timeout': ['off'],
            'playwright/no-nested-step': ['off'],
        },
    },
];
