// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import { screen, waitFor } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render, renderHook } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { ReadOnlyAnnotatorProviders } from './read-only-annotator-providers.component';

const mediaItem = getMockedMediaImage();

describe('ReadOnlyAnnotatorProviders', () => {
    beforeEach(() => {
        server.use(http.get('/api/projects/{project_id}', () => HttpResponse.json(getMockedProject())));
    });

    it('renders children within the provider tree', async () => {
        render(
            <ReadOnlyAnnotatorProviders mediaItem={mediaItem} initialAnnotationsDTO={[]} isUserReviewed={false}>
                <span>child content</span>
            </ReadOnlyAnnotatorProviders>
        );

        expect(await screen.findByText('child content')).toBeInTheDocument();
    });

    it('provides AnnotationActions with isReadOnlyMode true', async () => {
        const wrapper = ({ children }: { children: ReactNode }) => (
            <ReadOnlyAnnotatorProviders mediaItem={mediaItem} initialAnnotationsDTO={[]} isUserReviewed={false}>
                {children}
            </ReadOnlyAnnotatorProviders>
        );

        const { result } = renderHook(() => useAnnotationActions(), { wrapper });

        await waitFor(() => expect(result.current.isReadOnlyMode).toBe(true));
    });

    it('exposes annotations from initialAnnotationsDTO', async () => {
        const label = getMockedLabel({ id: 'label-1' });

        server.use(
            http.get('/api/projects/{project_id}', () =>
                HttpResponse.json(
                    getMockedProject({
                        task: { task_type: 'detection', exclusive_labels: true, labels: [label] },
                    })
                )
            )
        );

        const initialAnnotationsDTO = [
            {
                shape: { type: 'rectangle' as const, x: 10, y: 20, width: 100, height: 50 },
                labels: [{ id: label.id }],
            },
        ];

        const wrapper = ({ children }: { children: ReactNode }) => (
            <ReadOnlyAnnotatorProviders
                mediaItem={mediaItem}
                initialAnnotationsDTO={initialAnnotationsDTO}
                isUserReviewed={false}
            >
                {children}
            </ReadOnlyAnnotatorProviders>
        );

        const { result } = renderHook(() => useAnnotationActions(), { wrapper });

        await waitFor(() => {
            expect(result.current.annotations).toHaveLength(1);
            expect(result.current.annotations[0].shape).toEqual({
                type: 'rectangle',
                x: 10,
                y: 20,
                width: 100,
                height: 50,
            });
        });
    });
});
