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

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
const renderComponent = ({
    activeModelArchitectureId = undefined,
    selectedModelArchitectureId = null,
    onSelectedModelArchitectureIdChange = vi.fn(),
    modelArchitecture = getMockedModelArchitecture(),
}: {
    activeModelArchitectureId?: string;
    selectedModelArchitectureId?: string | null;
    onSelectedModelArchitectureIdChange?: ReturnType<typeof vi.fn>;
    modelArchitecture?: ModelArchitectureWithPerformanceCategory;
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
            />
        </ModelArchitecturesListLayout>
    );

    return { modelArchitecture };
};

describe('ModelArchitecture', () => {
    it('renders the model architecture name and number of parameters', () => {
        const modelArchitecture = getMockedModelArchitecture({ name: 'Deim-DFine-L' });
        renderComponent({ modelArchitecture });

        expect(screen.getByText(modelArchitecture.name)).toBeVisible();
        expect(screen.getByText(`${modelArchitecture.stats.trainable_parameters} million`)).toBeVisible();
    });

    it('does not render the "Active model" badge when the architecture is not active', () => {
        renderComponent({
            activeModelArchitectureId: 'some-other-id',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.queryByText('Active model')).not.toBeInTheDocument();
    });

    it('renders the "Active model" badge when the architecture is active', () => {
        renderComponent({
            activeModelArchitectureId: 'Object_Detection_Deim_DFine_L',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByText('Active model')).toBeVisible();
    });

    it('does not render a performance category badge when performanceCategory is undefined', () => {
        renderComponent({ modelArchitecture: getMockedModelArchitecture({ performanceCategory: undefined }) });

        expect(screen.queryByText(/speed|balance|accuracy/i)).not.toBeInTheDocument();
    });

    it('renders the performance category badge when performanceCategory is defined', () => {
        renderComponent({ modelArchitecture: getMockedModelArchitecture({ performanceCategory: 'speed' }) });

        expect(screen.getByText('Speed')).toBeVisible();
    });

    it('renders both "Active model" badge and performance category badge when both apply', () => {
        renderComponent({
            activeModelArchitectureId: 'Object_Detection_Deim_DFine_L',
            modelArchitecture: getMockedModelArchitecture({ performanceCategory: 'balance' }),
        });

        expect(screen.getByText('Active model')).toBeVisible();
        expect(screen.getByText('Balance')).toBeVisible();
    });

    it('calls onSelectedModelArchitectureIdChange with the architecture id when clicked', async () => {
        const user = userEvent.setup();
        const onSelectedModelArchitectureIdChange = vi.fn();

        renderComponent({ onSelectedModelArchitectureIdChange });

        await user.click(screen.getByText('Deim-DFine-L'));

        expect(onSelectedModelArchitectureIdChange).toHaveBeenCalledWith('Object_Detection_Deim_DFine_L');
    });

    it('renders the radio button as selected when selectedModelArchitectureId matches', () => {
        renderComponent({
            selectedModelArchitectureId: 'Object_Detection_Deim_DFine_L',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).toBeChecked();
    });

    it('renders the radio button as not selected when selectedModelArchitectureId does not match', () => {
        renderComponent({
            selectedModelArchitectureId: 'some-other-id',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).not.toBeChecked();
    });

    it('renders the radio button as not selected when selectedModelArchitectureId is null', () => {
        renderComponent({ selectedModelArchitectureId: null });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).not.toBeChecked();
    });
});
