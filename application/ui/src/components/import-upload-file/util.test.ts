// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isSupportedDatasetZip } from './util';

describe('import-upload-file utils', () => {
    describe('isSupportedDatasetZip', () => {
        it('return true for zip files', () => {
            const file = new File([''], 'dataset.zip', { type: 'application/zip' });
            expect(isSupportedDatasetZip(file)).toBe(true);
        });

        it('return true for zip files (windows)', () => {
            const file = new File([''], 'dataset.zip', { type: 'application/x-zip-compressed' });
            expect(isSupportedDatasetZip(file)).toBe(true);
        });

        it('return false for non-zip files', () => {
            const file = new File([''], 'image.png', { type: 'image/png' });
            expect(isSupportedDatasetZip(file)).toBe(false);
        });

        it('return false for files without type', () => {
            const file = new File([''], 'unknownfile');
            expect(isSupportedDatasetZip(file)).toBe(false);
        });
    });
});
