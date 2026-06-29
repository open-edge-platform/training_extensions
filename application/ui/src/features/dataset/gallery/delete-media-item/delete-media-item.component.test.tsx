// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { DeleteMediaItem } from './delete-media-item.component';

describe('DeleteMediaItem', () => {
    it('deletes a single media item and shows a success toast', async () => {
        const itemId = '123';
        const mockedOnDeleted = vitest.fn();
        let requestBody: { media_ids?: string[] } | undefined;

        server.use(
            http.delete('/api/projects/{project_id}/dataset/media', async ({ request }) => {
                requestBody = (await request.json()) as { media_ids: string[] };
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<DeleteMediaItem itemsIds={[itemId]} onDeleted={mockedOnDeleted} />);

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        expect(await screen.findByText(/Are you sure you want to delete 1 item\?/i)).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`1 item deleted successfully`)).toBeVisible();
        expect(requestBody).toEqual({ media_ids: [itemId] });
        expect(mockedOnDeleted).toHaveBeenCalledWith([itemId]);
    });

    it('deletes multiple media items and shows a success toast', async () => {
        const itemsIds = ['123', '456', '789'];
        const mockedOnDeleted = vitest.fn();
        let requestBody: { media_ids?: string[] } | undefined;

        server.use(
            http.delete('/api/projects/{project_id}/dataset/media', async ({ request }) => {
                requestBody = (await request.json()) as { media_ids: string[] };
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<DeleteMediaItem itemsIds={itemsIds} onDeleted={mockedOnDeleted} />);

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        expect(await screen.findByText(/Are you sure you want to delete 3 items\?/i)).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`3 items deleted successfully`)).toBeVisible();
        expect(requestBody).toEqual({ media_ids: itemsIds });
        expect(mockedOnDeleted).toHaveBeenCalledWith(itemsIds);
    });

    it('shows an error toast when deleting media items fails', async () => {
        const itemsIds = ['123', '456'];
        const errorMessage = 'test error message';
        const mockedOnDeleted = vitest.fn();

        server.use(
            http.delete('/api/projects/{project_id}/dataset/media', () => {
                // @ts-expect-error error response schema
                return HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        render(<DeleteMediaItem itemsIds={itemsIds} onDeleted={mockedOnDeleted} />);

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(/Are you sure you want to delete 2 items\?/i);

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`Failed to delete, ${errorMessage}`)).toBeVisible();
        expect(mockedOnDeleted).not.toHaveBeenCalled();
    });
});
