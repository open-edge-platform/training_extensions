// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { render } from 'test-utils/render';

import { useDownloadModel } from '../../hooks/api/use-download-model.hook';
import { ModelVariantTable } from './model-variant-table.component';

vi.mock('../../hooks/api/use-download-model.hook', () => ({
    useDownloadModel: vi.fn(),
}));

const mockDownloadModel = vi.fn();

describe('ModelVariantTable', () => {
    beforeEach(() => {
        vi.clearAllMocks();

        vi.mocked(useDownloadModel).mockReturnValue({
            downloadModel: mockDownloadModel,
            isDownloading: false,
            error: null,
        });
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

        expect(screen.getByText('Accuracy')).toBeInTheDocument();
        expect(screen.getByText('92%')).toBeInTheDocument();
    });

    it('falls back to fp32 pytorch primary metric when current variant evaluations are empty', () => {
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

        expect(screen.getByText('mAP')).toBeInTheDocument();
        expect(screen.getByText('87%')).toBeInTheDocument();
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

        expect(screen.getByText('mAP')).toBeInTheDocument();
        expect(screen.getByText('-')).toBeInTheDocument();
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

        expect(screen.getByText('Score')).toBeInTheDocument();
        expect(screen.getByText('-')).toBeInTheDocument();
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

        expect(mockDownloadModel).toHaveBeenCalledWith('ov-1');
    });
});
