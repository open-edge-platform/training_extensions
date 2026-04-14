// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModelArchitecture } from 'mocks/mock-model';
import { render } from 'test-utils/render';

import { ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';
import { ModelArchitecture } from './model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

const renderModelArchitecture = ({
    activeModelArchitectureId = undefined,
    selectedModelArchitectureId = null,
    onSelectedModelArchitectureIdChange = vi.fn(),
    modelArchitecture = getMockedModelArchitecture(),
    showBenchmarkStats = false,
}: {
    activeModelArchitectureId?: string;
    selectedModelArchitectureId?: string | null;
    onSelectedModelArchitectureIdChange?: ReturnType<typeof vi.fn>;
    modelArchitecture?: ModelArchitectureWithPerformanceCategory;
    showBenchmarkStats?: boolean;
} = {}) => {
    render(
        <ModelArchitecturesListLayout
            selectedModelArchitectureId={selectedModelArchitectureId}
            onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            ariaLabel={'Model architectures'}
        >
            <ModelArchitecture
                activeModelArchitectureId={activeModelArchitectureId}
                modelArchitecture={modelArchitecture}
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                showBenchmarkStats={showBenchmarkStats}
            />
        </ModelArchitecturesListLayout>
    );

    return { modelArchitecture };
};

describe('ModelArchitecture', () => {
    it('renders the model architecture name and number of parameters', () => {
        const modelArchitecture = getMockedModelArchitecture({ name: 'Deim-DFine-L' });
        renderModelArchitecture({ modelArchitecture });

        expect(screen.getByText(modelArchitecture.name)).toBeVisible();
        expect(
            screen.getByText(`Number of parameters: ${modelArchitecture.stats.trainable_parameters} million`)
        ).toBeVisible();
    });

    it('does not render the "Active model" badge when the architecture is not active', () => {
        renderModelArchitecture({
            activeModelArchitectureId: 'some-other-id',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.queryByText('Active model')).not.toBeInTheDocument();
    });

    it('renders the "Active model" badge when the architecture is active', () => {
        renderModelArchitecture({
            activeModelArchitectureId: 'Object_Detection_Deim_DFine_L',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByText('Active model')).toBeVisible();
    });

    it('does not render a performance category badge when performanceCategory is undefined', () => {
        renderModelArchitecture({ modelArchitecture: getMockedModelArchitecture({ performanceCategory: undefined }) });

        expect(screen.queryByText(/speed|balance|accuracy/i)).not.toBeInTheDocument();
    });

    it('renders the performance category badge when performanceCategory is defined', () => {
        renderModelArchitecture({ modelArchitecture: getMockedModelArchitecture({ performanceCategory: 'speed' }) });

        expect(screen.getByText('Speed')).toBeVisible();
    });

    it('renders both "Active model" badge and performance category badge when both apply', () => {
        renderModelArchitecture({
            activeModelArchitectureId: 'Object_Detection_Deim_DFine_L',
            modelArchitecture: getMockedModelArchitecture({ performanceCategory: 'balance' }),
        });

        expect(screen.getByText('Active model')).toBeVisible();
        expect(screen.getByText('Balance')).toBeVisible();
    });

    it('calls onSelectedModelArchitectureIdChange with the architecture id when clicked', async () => {
        const user = userEvent.setup();
        const onSelectedModelArchitectureIdChange = vi.fn();

        renderModelArchitecture({ onSelectedModelArchitectureIdChange });

        await user.click(screen.getByText('Deim-DFine-L'));

        expect(onSelectedModelArchitectureIdChange).toHaveBeenCalledWith('Object_Detection_Deim_DFine_L');
    });

    it('renders the radio button as selected when selectedModelArchitectureId matches', () => {
        renderModelArchitecture({
            selectedModelArchitectureId: 'Object_Detection_Deim_DFine_L',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).toBeChecked();
    });

    it('renders the radio button as not selected when selectedModelArchitectureId does not match', () => {
        renderModelArchitecture({
            selectedModelArchitectureId: 'some-other-id',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).not.toBeChecked();
    });

    it('renders the radio button as not selected when selectedModelArchitectureId is null', () => {
        renderModelArchitecture({ selectedModelArchitectureId: null });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).not.toBeChecked();
    });

    describe('showBenchmarkStats', () => {
        const detectionArchitecture = getMockedModelArchitecture({
            task: 'detection',
            stats: {
                gigaflops: 91,
                trainable_parameters: 31,
                benchmark_metrics: {
                    imagenet_top1_accuracy: null,
                    imagenet_top5_accuracy: null,
                    coco_map_50_95: 55.3,
                    coco_map_50: null,
                },
            },
            performanceCategory: 'speed',
        });

        it('does not show gigaflops or accuracy by default', () => {
            renderModelArchitecture({ modelArchitecture: detectionArchitecture });

            expect(screen.queryByText(/gigaflops/i)).not.toBeInTheDocument();
            expect(screen.queryByText(/mAP/i)).not.toBeInTheDocument();
        });

        it('shows gigaflops and mAP when showBenchmarkStats is true', () => {
            renderModelArchitecture({ modelArchitecture: detectionArchitecture, showBenchmarkStats: true });

            expect(screen.getByText(`Gigaflops: ${detectionArchitecture.stats.gigaflops}`)).toBeVisible();
            expect(screen.getByText('mAP: 55.3%')).toBeVisible();
        });

        it('hides the performance category badge when showBenchmarkStats is true', () => {
            renderModelArchitecture({ modelArchitecture: detectionArchitecture, showBenchmarkStats: true });

            expect(screen.queryByText('Speed')).not.toBeInTheDocument();
        });

        it('still shows the performance category badge when showBenchmarkStats is false', () => {
            renderModelArchitecture({ modelArchitecture: detectionArchitecture, showBenchmarkStats: false });

            expect(screen.getByText('Speed')).toBeVisible();
        });

        it('shows Top-1 Acc for classification tasks', () => {
            const classificationArchitecture = getMockedModelArchitecture({
                task: 'classification',
                stats: {
                    gigaflops: 2,
                    trainable_parameters: 5,
                    benchmark_metrics: {
                        imagenet_top1_accuracy: 76.2,
                        imagenet_top5_accuracy: 95.3,
                        coco_map_50_95: null,
                        coco_map_50: null,
                    },
                },
            });

            renderModelArchitecture({ modelArchitecture: classificationArchitecture, showBenchmarkStats: true });

            expect(screen.getByText('Top-1 Acc: 76.2%')).toBeVisible();
        });
    });
});
