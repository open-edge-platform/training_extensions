// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { TotalItems } from './total-items.component';

describe('TotalItems', () => {
    const renderTotalItems = async (totalSelectedElements: number, totalItems: number) => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: [getMockedMediaImage({})],
                    pagination: { offset: 0, limit: 1, count: totalItems, total: totalItems },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    pagination: {
                        total: totalItems,
                        offset: 0,
                        limit: 0,
                        count: 0,
                    },
                    items: [],
                });
            })
        );

        return render(<TotalItems totalSelectedElements={totalSelectedElements} />);
    };

    it('shows selected count when items are selected', async () => {
        await renderTotalItems(3, 10);

        expect(screen.getByText('3 selected')).toBeVisible();
    });

    it('shows total media count with plural when no items are selected', async () => {
        await renderTotalItems(0, 5);

        expect(await screen.findByText('5 media items')).toBeVisible();
    });

    it('shows singular media when there is exactly 1 item', async () => {
        await renderTotalItems(0, 1);

        expect(await screen.findByText('1 media item')).toBeVisible();
    });

    it('shows 0 media items when there are no items', async () => {
        await renderTotalItems(0, 0);

        expect(await screen.findByText('0 media items')).toBeVisible();
    });
});
