// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { DatasetStatistics } from './dataset-statistics.component';

describe('DatasetStatistics', () => {
    const renderApp = ({ totalItems = 10, annotatedItems = 0 }: { totalItems?: number; annotatedItems?: number }) => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items', ({ request }) => {
                const url = new URL(request.url);
                const annotationStatus = url.searchParams.get('annotation_status');

                if (annotationStatus === 'reviewed') {
                    return HttpResponse.json({
                        pagination: { total: annotatedItems, offset: 0, limit: 0, count: 0 },
                        items: [],
                    });
                }

                return HttpResponse.json({
                    pagination: { total: totalItems, offset: 0, limit: 0, count: 0 },
                    items: [],
                });
            })
        );

        render(<DatasetStatistics />);
    };

    it('displays correct statistics when all items are annotated', async () => {
        renderApp({ totalItems: 100, annotatedItems: 33 });

        expect(await screen.findByText('100')).toBeVisible();

        expect(screen.getByText('Annotated')).toBeVisible();
        expect(screen.getByText('33%')).toBeVisible();
        expect(screen.getByText('33 images')).toBeVisible();

        expect(screen.getByText('Unannotated')).toBeVisible();
        expect(screen.getByText('67%')).toBeVisible();
        expect(screen.getByText('67 images')).toBeVisible();
    });
});
