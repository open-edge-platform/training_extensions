// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import { act, waitFor } from '@testing-library/react';
import { getMockedShape } from 'mocks/mock-annotation';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http } from '../../api/utils';
import type { AnnotationDTO, Label } from '../../constants/shared-types';
import { server } from '../../msw-node-setup';
import { renderHook } from '../../test-utils/render';
import {
    AnnotationActionsProvider,
    syncAnnotationLabelsWithProjectLabels,
    useAnnotationActions,
    type AnnotationActionsProviderProps,
} from './annotation-actions-provider.component';
import type { AnnotatorMode } from './annotator-mode';
import { EMPTY_LABEL_ID } from './labels';

const renderAnnotationActions = ({
    initialAnnotationsDTO = [],
    initialPredictionsDTO = [],
    mediaItem = getMockedMediaImage(),
    mode = 'prediction' as AnnotatorMode,
    isReadOnly = false,
    isUserReviewed = false,
    labels = [],
}: Partial<AnnotationActionsProviderProps> & {
    labels: Label[];
}) => {
    server.use(
        http.get('/api/projects/{project_id}', () =>
            HttpResponse.json(getMockedProject({ task: { task_type: 'detection', exclusive_labels: false, labels } }))
        )
    );

    const wrapper = ({ children }: { children: ReactNode }) => (
        <AnnotationActionsProvider
            mode={mode}
            mediaItem={mediaItem}
            isReadOnly={isReadOnly}
            initialAnnotationsDTO={initialAnnotationsDTO}
            initialPredictionsDTO={initialPredictionsDTO}
            isUserReviewed={isUserReviewed}
        >
            {children}
        </AnnotationActionsProvider>
    );

    return renderHook(() => useAnnotationActions(), { wrapper });
};

describe('submitPredictions', () => {
    const label1 = getMockedLabel({ id: 'label-1', name: 'Cat', color: '#FF0000' });
    const label2 = getMockedLabel({ id: 'label-2', name: 'Dog', color: '#00FF00' });

    it('submits predictions with confidences stripped', async () => {
        const predictionsDTO: AnnotationDTO[] = [
            {
                labels: [{ id: label1.id }],
                shape: getMockedShape({ type: 'rectangle' }),
                confidences: [0.95],
            },
        ];

        let savedBody: { annotations: AnnotationDTO[]; subset?: string } | undefined;

        server.use(
            http.post('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ request }) => {
                savedBody = (await request.json()) as typeof savedBody;
                return HttpResponse.json({});
            })
        );

        const { result } = renderAnnotationActions({
            labels: [label1, label2],
            initialPredictionsDTO: predictionsDTO,
            mode: 'prediction',
        });

        await waitFor(() => expect(result.current).not.toBeNull());

        await act(async () => {
            await result.current.submitPredictions('training');
        });

        await waitFor(() => {
            expect(savedBody).toBeDefined();
            expect(savedBody?.annotations).toHaveLength(1);
            expect(savedBody?.annotations[0]).not.toHaveProperty('confidences');
            expect(savedBody?.annotations[0].labels).toEqual([{ id: label1.id }]);
        });
    });

    it('filters out annotations with empty labels', async () => {
        const predictionsDTO: AnnotationDTO[] = [
            { labels: [{ id: EMPTY_LABEL_ID }], shape: { type: 'full_image' } },
            { labels: [{ id: label1.id }], shape: getMockedShape({ type: 'rectangle' }) },
        ];

        let savedBody: { annotations: AnnotationDTO[] } | undefined;

        server.use(
            http.post('/api/projects/{project_id}/dataset/media/{media_id}/annotations', async ({ request }) => {
                savedBody = (await request.json()) as typeof savedBody;
                return HttpResponse.json({});
            })
        );

        const { result } = renderAnnotationActions({
            mode: 'prediction',
            labels: [label1, label2],
            initialPredictionsDTO: predictionsDTO,
        });

        await waitFor(() => expect(result.current).not.toBeNull());

        await act(() => result.current.submitPredictions('training'));

        await waitFor(() => {
            expect(savedBody).toBeDefined();
            expect(savedBody?.annotations).toHaveLength(1);
            expect(savedBody?.annotations[0].labels).toEqual([{ id: label1.id }]);
        });
    });
});

describe('Label synchronization', () => {
    const sourceLabel = getMockedLabel({ id: 'label-1', name: 'Fire', color: '#0000FF', hotkey: 'F' });

    it('updates annotation label metadata when project label changes', () => {
        const annotations = [
            {
                id: 'annotation-1',
                shape: getMockedShape({ type: 'rectangle' }),
                labels: [sourceLabel],
            },
        ];

        const updatedLabel = getMockedLabel({ id: 'label-1', name: 'Fire23', color: '#FF0000', hotkey: 'R' });

        const result = syncAnnotationLabelsWithProjectLabels(annotations, [updatedLabel]);

        expect(result[0].labels).toEqual([updatedLabel]);
    });

    it('removes annotation labels that no longer exist in project labels', () => {
        const annotations = [
            {
                id: 'annotation-1',
                shape: getMockedShape({ type: 'rectangle' }),
                labels: [sourceLabel],
            },
        ];

        const result = syncAnnotationLabelsWithProjectLabels(annotations, []);

        expect(result[0].labels).toEqual([]);
    });
});
