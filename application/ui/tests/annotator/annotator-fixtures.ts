// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { getMockedLabel } from 'mocks/mock-labels';
import { HttpResponse } from 'msw';

import { http } from '../fixtures';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

export const candyPngBuffer = fs.readFileSync(path.resolve(dirname, '../assets/candy.png'));

export const redLabel = getMockedLabel({ id: 'red-label', name: 'red-label', color: '#ad2323' });
export const blueLabel = getMockedLabel({ id: 'blue-label', name: 'blue-label', color: '#2424a0' });

export const candyBinaryHandler = http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', async () => {
    return HttpResponse.arrayBuffer(candyPngBuffer.buffer, {
        headers: { 'Content-Type': 'image/png' },
    });
});
