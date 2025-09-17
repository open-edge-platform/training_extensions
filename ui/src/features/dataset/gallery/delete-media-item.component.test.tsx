// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitForElementToBeRemoved } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { getMockedMediaItem } from '../../../../tests/test-utils/mocks';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { TestProviders } from '../../../providers';
import { DeleteMediaItem } from './delete-media-item.component';

vi.mock('react-router', () => ({
    useParams: vi.fn(() => ({ projectId: '123' })),
}));

describe('DeleteMediaItem', () => {
    const mockedItem = getMockedMediaItem({});

    it('deletes a media item and shows a success toast', async () => {
        server.use(
            http.delete('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(
            <TestProviders>
                <DeleteMediaItem item={mockedItem} />
            </TestProviders>
        );

        userEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(`Are you sure you want to delete the item "${mockedItem.id}"?`);

        userEvent.click(screen.getByRole('button', { name: /confirm/i }));
        await waitForElementToBeRemoved(() => screen.queryByRole('button', { name: /confirm/i }));

        expect(screen.getByLabelText('toast')).toHaveTextContent(`Item "${mockedItem.id}" was deleted successfully`);
    });

    it('shows an error toast when deleting a media item fails', async () => {
        const errorMessage = 'test error message';
        server.use(
            http.delete('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                // @ts-ignore
                return HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        render(
            <TestProviders>
                <DeleteMediaItem item={mockedItem} />
            </TestProviders>
        );

        userEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(`Are you sure you want to delete the item "${mockedItem.id}"?`);

        userEvent.click(screen.getByRole('button', { name: /confirm/i }));
        await waitForElementToBeRemoved(() => screen.queryByRole('button', { name: /confirm/i }));

        expect(screen.getByLabelText('toast')).toHaveTextContent(`Failed to delete item: ${errorMessage}`);
    });
});
