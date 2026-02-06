// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { render } from 'test-utils/render';

import { getMockedModel } from '../../../../../../mocks/mock-model';
import { ModelRow } from './model-row.component';

describe('ModelRow', () => {
    const defaultModel = getMockedModel({
        id: 'model-123',
        name: 'Test Model',
        architecture: 'YOLOX',
        size: 1024,
        training_info: {
            status: 'successful',
            label_schema_revision: {
                labels: [],
            },
            start_time: '2025-01-10T10:00:00.000000+00:00',
            end_time: '2025-01-10T12:30:00.000000+00:00',
            dataset_revision_id: 'dataset-123',
        },
    });

    const datasetRevision = getMockedDatasetRevision({
        id: 'dataset-123',
        name: 'Dataset 1',
        item_counts: {
            total: 10,
            testing: 4,
            training: 4,
            validation: 2,
        },
    });

    describe('basic rendering', () => {
        it('should render all model information correctly when grouped by architecture', () => {
            render(<ModelRow model={defaultModel} datasetRevision={datasetRevision} groupBy={'architecture'} />);

            expect(screen.getByTestId('model-name')).toHaveTextContent('Test Model');

            const datasetBadge = screen.getByTestId('dataset-count');
            const labelsBadge = screen.getByTestId('labels-count');
            const labelSchemaRevision = defaultModel.training_info.label_schema_revision ?? {};
            const labelsCount =
                'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
                    ? labelSchemaRevision.labels.length
                    : '';

            expect(screen.getByText(datasetRevision.name)).toBeInTheDocument();
            expect(within(datasetBadge).getByText(datasetRevision.item_counts?.total?.toString() ?? ''));
            expect(within(labelsBadge).getByText(labelsCount));
            expect(screen.queryByText(/YOLOX/)).not.toBeInTheDocument();
            expect(screen.queryByText('Speed')).not.toBeInTheDocument();
        });

        it('should render all model information correctly when grouped by dataset', () => {
            render(<ModelRow model={defaultModel} datasetRevision={datasetRevision} groupBy={'dataset'} />);

            expect(screen.getByTestId('model-name')).toHaveTextContent('Test Model');

            expect(screen.getByText(/YOLOX/)).toBeInTheDocument();
            expect(screen.getByText('Speed')).toBeInTheDocument();
            expect(screen.queryByText(datasetRevision.name)).not.toBeInTheDocument();
            expect(screen.queryByTestId('dataset-count')).not.toBeInTheDocument();
            expect(screen.queryByTestId('labels-count')).not.toBeInTheDocument();
        });

        it('should render "Unnamed Model" when model name is null or undefined', () => {
            const modelWithoutName = getMockedModel({ name: undefined });

            render(<ModelRow model={modelWithoutName} datasetRevision={datasetRevision} groupBy={'dataset'} />);

            expect(screen.getByTestId('model-name')).toHaveTextContent('Unnamed Model');
        });

        it('should render "-" when model size is 0 or negative', () => {
            const modelWithZeroSize = getMockedModel({ size: 0 });

            render(<ModelRow model={modelWithZeroSize} datasetRevision={datasetRevision} groupBy={'dataset'} />);

            expect(screen.getByText('-')).toBeInTheDocument();
        });

        it('renders "Failed" badge when training status is failed', () => {
            const failedModel = getMockedModel({
                training_info: {
                    status: 'failed',
                },
            });

            render(<ModelRow model={failedModel} datasetRevision={datasetRevision} groupBy={'dataset'} />);

            expect(screen.getByText('Failed')).toBeInTheDocument();
        });
    });

    describe('active model tag', () => {
        it('should show active tag only when model id matches activeModelArchitectureId', () => {
            const { rerender } = render(
                <ModelRow
                    model={defaultModel}
                    activeModelId='model-123'
                    datasetRevision={datasetRevision}
                    groupBy={'dataset'}
                />
            );
            expect(screen.getByText('Active')).toBeInTheDocument();

            rerender(
                <ModelRow
                    model={defaultModel}
                    activeModelId={'different-id'}
                    datasetRevision={datasetRevision}
                    groupBy={'dataset'}
                />
            );
            expect(screen.queryByText('Active')).not.toBeInTheDocument();

            rerender(<ModelRow model={defaultModel} datasetRevision={datasetRevision} groupBy={'dataset'} />);
            expect(screen.queryByText('Active')).not.toBeInTheDocument();
        });
    });

    describe('parent revision model', () => {
        it('should render parent revision model when provided and call onExpandModel when clicked', async () => {
            const onExpandModel = vi.fn();
            const parentModel = getMockedModel({
                id: 'parent-123',
                name: 'Parent Model',
            });

            render(
                <ModelRow
                    model={defaultModel}
                    parentRevisionModel={parentModel}
                    onExpandModel={onExpandModel}
                    datasetRevision={datasetRevision}
                    groupBy={'dataset'}
                />
            );

            expect(screen.getByText('Fine-tuned from')).toBeInTheDocument();
            const parentLink = screen.getByRole('link', { name: 'Parent Model' });
            expect(parentLink).toBeInTheDocument();

            await userEvent.click(parentLink);
            expect(onExpandModel).toHaveBeenCalledWith('parent-123');
        });

        it('should not render parent revision model when not provided', () => {
            render(<ModelRow model={defaultModel} datasetRevision={datasetRevision} groupBy={'dataset'} />);

            expect(screen.queryByText('Fine-tuned from')).not.toBeInTheDocument();
        });
    });
});
