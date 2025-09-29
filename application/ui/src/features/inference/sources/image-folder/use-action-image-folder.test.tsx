// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { startTransition } from 'react';

import { screen, TestProviders } from '@test-utils/render';
import { act, renderHook, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { ImagesFolderSourceConfig } from '../util';
import { useActionImageFolder } from './use-action-image-folder.hook';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

const mockedConfig: ImagesFolderSourceConfig = {
    name: 'Test Folder',
    source_type: 'images_folder',
    images_folder_path: '/path/to/images',
    ignore_existing_images: false,
};

const renderApp = async ({
    isNew = false,
    config = mockedConfig,
    newResource = () => HttpResponse.error(),
    updateResource = () => HttpResponse.error(),
}) => {
    server.use(
        http.post('/api/sources', newResource),
        http.patch('/api/sources/{source_id}', updateResource),
        http.patch('/api/projects/{project_id}/pipeline', () => HttpResponse.json({}))
    );

    const { result } = renderHook(() => useActionImageFolder(config, isNew), { wrapper: TestProviders });
    const [_state, submitAction] = result.current;

    const formData = new FormData();
    formData.append('name', config.name);
    formData.append('source_type', config.source_type);
    formData.append('images_folder_path', config.images_folder_path);
    formData.append('ignore_existing_images', String(config.ignore_existing_images));
    config.id && formData.append('id', config.id);

    await act(async () => {
        startTransition(async () => submitAction(formData));
    });

    return result.current;
};

describe('useActionImageFolder', () => {
    it('return initial config', () => {
        const { result } = renderHook(() => useActionImageFolder(mockedConfig), { wrapper: TestProviders });

        expect(result.current[0]).toEqual(mockedConfig);
    });

    describe('new configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';
            await renderApp({
                isNew: true,
                newResource: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save source configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submit new image folder config and show success message', async () => {
            const mockedNewItemId = 'new-id-test';
            const [state] = await renderApp({
                isNew: true,
                newResource: () => HttpResponse.json({ ...mockedConfig, id: mockedNewItemId }),
            });

            await waitFor(() => {
                expect(screen.getByText('Image folder configuration created successfully.')).toBeVisible();
                expect(state.id).toBe(mockedNewItemId);
            });
        });
    });

    describe('edit configuration', () => {
        it('submits folder config and display error message on failure', async () => {
            const mockedError = 'test-error';

            await renderApp({
                isNew: false,
                updateResource: () => HttpResponse.json({ detail: mockedError }, { status: 400 }),
            });

            await waitFor(() => {
                expect(screen.getByText(`Failed to save source configuration, ${mockedError}`)).toBeVisible();
            });
        });

        it('submits folder config and show success message', async () => {
            const newConfig = { ...mockedConfig, id: 'mockedResponseId', name: 'Updated Name' };

            const [state] = await renderApp({
                isNew: false,
                config: newConfig,
                updateResource: () => HttpResponse.json(newConfig),
            });

            await waitFor(() => {
                expect(screen.getByText('Image folder configuration updated successfully.')).toBeVisible();
                expect(state.id).toBe(newConfig.id);
            });
        });
    });
});
