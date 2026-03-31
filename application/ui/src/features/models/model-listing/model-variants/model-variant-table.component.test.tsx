// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { render } from 'test-utils/render';

import { downloadFile } from '../../../../shared/util';
import { ModelVariantTable } from './model-variant-table.component';

vi.mock('../../../../shared/util', async (importOriginal) => ({
    ...(await importOriginal<typeof import('../../../../shared/util')>()),
    downloadFile: vi.fn(),
}));

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

describe('ModelVariantTable', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('shows primary testing metric value for a variant', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.923, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'Accuracy' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('92%');
    });

    it('falls back to fp32 pytorch primary metric when evaluations are empty', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [],
                }),
                getMockedVariant({
                    id: 'pt-1',
                    format: 'pytorch',
                    precision: 'fp32',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'mAP', value: 0.87, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'mAP' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('87%');
    });

    it('shows size and performance deltas for the quantized variant', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-fp16',
                    format: 'openvino',
                    precision: 'fp16',
                    weights_size: 1000,
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.92, primary: true }],
                        },
                    ],
                }),
                getMockedVariant({
                    id: 'ov-int8',
                    format: 'openvino',
                    precision: 'int8',
                    weights_size: 500,
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.89, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByText('-50%')).toHaveStyle({ color: 'var(--moss-tint-1)' });
        expect(screen.getByText('-3%')).toHaveStyle({ color: 'var(--coral-shade-1)' });
    });

    it('does not fallback value when variant has evaluations but no primary metric', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.91, primary: false }],
                        },
                    ],
                }),
                getMockedVariant({
                    id: 'pt-1',
                    format: 'pytorch',
                    precision: 'fp32',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'mAP', value: 0.87, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'mAP' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('-');
        expect(screen.queryByText('87%')).not.toBeInTheDocument();
    });

    it('shows Score and dash when no primary metric is available anywhere', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'Score' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('-');
    });

    it('triggers download for selected variant', async () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        await userEvent.click(screen.getByRole('button', { name: 'Download model ov-1' }));

        expect(downloadFile).toHaveBeenCalledWith(
            expect.stringContaining(`/api/projects/project-123/models/${model.id}/variants/ov-1/binary`)
        );
    });
});
