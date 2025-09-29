// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitForElementToBeRemoved } from '@test-utils/render';
import { userEvent } from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { vi } from 'vitest';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { DeleteMediaItem } from './delete-media-item.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

describe('DeleteMediaItem', () => {
    it('deletes a media item and shows a success toast', async () => {
        const itemId = '123';
        const mockedOnDeleted = vitest.fn();

        server.use(
            http.delete('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<DeleteMediaItem itemsIds={[itemId]} onDeleted={mockedOnDeleted} />);

        userEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(/Are you sure you want to delete the next items?/i);

        userEvent.click(screen.getByRole('button', { name: /confirm/i }));
        await waitForElementToBeRemoved(() => screen.queryByRole('button', { name: /confirm/i }));

        expect(screen.getByText(`1 item(s) deleted successfully`)).toBeVisible();
        expect(mockedOnDeleted).toHaveBeenCalledWith([itemId]);
    });

    it('shows an error toast when deleting a media item fails', async () => {
        const itemToFail = '321';
        const itemToDelete = '123';
        const errorMessage = 'test error message';
        const mockedOnDeleted = vitest.fn();

        server.use(
            http.delete('/api/projects/{project_id}/dataset/items/{dataset_item_id}', ({ params }) => {
                const { dataset_item_id } = params;
                return dataset_item_id === itemToDelete
                    ? HttpResponse.json(null, { status: 204 })
                    : // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                      // @ts-expect-error
                      HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        render(<DeleteMediaItem itemsIds={[itemToFail, itemToDelete]} onDeleted={mockedOnDeleted} />);

        userEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(/Are you sure you want to delete the next items?/i);

        userEvent.click(screen.getByRole('button', { name: /confirm/i }));
        await waitForElementToBeRemoved(() => screen.queryByRole('button', { name: /confirm/i }));

        expect(screen.getByText(`1 item(s) deleted successfully`)).toBeVisible();
        expect(screen.getByText(`Failed to delete, ${errorMessage}`)).toBeVisible();
        expect(mockedOnDeleted).toHaveBeenCalledWith([itemToDelete]);
    });
});
