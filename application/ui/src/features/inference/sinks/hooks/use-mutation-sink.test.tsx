// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { TestProviders } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { LocalFolderSinkConfig } from '../utils';
import { useMutationSink } from './use-mutation-sink.hook';

const mockedSource: LocalFolderSinkConfig = {
    id: 'original-id',
    name: 'Mock Source',
    output_formats: [],
    sink_type: 'folder',
    folder_path: './folder/111',
};

describe('useMutationSink', () => {
    it('creates a new sink and return its resource id', async () => {
        const newResourceId = 'resource-id-123';
        const { result } = renderHook(() => useMutationSink(true), {
            wrapper: TestProviders,
        });

        server.use(
            http.post('/api/sinks', () => HttpResponse.json({ ...mockedSource, id: newResourceId })),
            http.patch('/api/sinks/{sink_id}', () => HttpResponse.error())
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(newResourceId);
        });
    });

    it('update a source item and returns its resource id', async () => {
        const { result } = renderHook(() => useMutationSink(false), {
            wrapper: TestProviders,
        });

        server.use(
            http.post('/api/sinks', () => HttpResponse.error()),
            http.patch('/api/sinks/{sink_id}', () => HttpResponse.json(mockedSource))
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(mockedSource.id);
        });
    });
});
