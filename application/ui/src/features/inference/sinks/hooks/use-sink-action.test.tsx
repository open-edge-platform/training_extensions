// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { startTransition } from 'react';

import { act, renderHook, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { screen, TestProviders } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { LocalFolderSinkConfig, SinkOutputFormats } from '../utils';
import { useSinkAction } from './use-sink-action.hook';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

const mockedConfig: LocalFolderSinkConfig = {
    id: '1',
    name: 'Test Folder',
    output_formats: [],
    sink_type: 'folder',
    folder_path: '',
    rate_limit: 0,
};

const bodyFormatter = (formData: FormData) => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    sink_type: 'folder' as const,
    rate_limit: formData.get('rate_limit') ? Number(formData.get('rate_limit')) : 0,
    folder_path: String(formData.get('folder_path')),
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
});

const renderApp = async ({
    isNewSink = false,
    config = mockedConfig,
    newSink = () => HttpResponse.error(),
    updateSink = () => HttpResponse.error(),
}) => {
    server.use(
        http.post('/api/sinks', newSink),
        http.patch('/api/sinks/{sink_id}', updateSink),
        http.patch('/api/projects/{project_id}/pipeline', () => HttpResponse.json({}))
    );

    const { result } = renderHook(() => useSinkAction({ config: mockedConfig, isNewSink, bodyFormatter }), {
        wrapper: TestProviders,
    });
    const [_state, submitAction] = result.current;

    const formData = new FormData();
    formData.append('name', config.name);
    formData.append('sink_type', config.sink_type);
    formData.append('rate_limit', String(config.rate_limit));
    formData.append('folder_path', config.folder_path);
    formData.append('output_formats', config.output_formats.join(','));

    config.id && formData.append('id', config.id);

    await act(async () => {
        startTransition(async () => submitAction(formData));
    });

    return result.current;
};

describe('useSinkAction', () => {
    it('return initial config', () => {
        const { result } = renderHook(() => useSinkAction({ config: mockedConfig, isNewSink: true, bodyFormatter }), {
            wrapper: TestProviders,
        });

        expect(result.current[0]).toEqual(mockedConfig);
    });

    describe('new configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';
            await renderApp({
                isNewSink: true,
                newSink: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save sink configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submit new image folder config and show success message', async () => {
            const mockedNewItemId = 'new-id-test';
            const [state] = await renderApp({
                isNewSink: true,
                newSink: () => HttpResponse.json({ ...mockedConfig, id: mockedNewItemId }),
            });

            await waitFor(() => {
                expect(screen.getByText('Sink configuration created successfully.')).toBeVisible();
                expect(state.id).toBe(mockedNewItemId);
            });
        });
    });

    describe('edit configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';

            await renderApp({
                isNewSink: false,
                updateSink: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save sink configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submits folder config and show success message', async () => {
            const newConfig = { ...mockedConfig, id: 'mockedResponseId', name: 'Updated Name' };

            const [state] = await renderApp({
                isNewSink: false,
                config: newConfig,
                updateSink: () => HttpResponse.json(newConfig),
            });

            await waitFor(() => {
                expect(screen.getByText('Sink configuration updated successfully.')).toBeVisible();
                expect(state.id).toBe(newConfig.id);
            });
        });
    });
});
