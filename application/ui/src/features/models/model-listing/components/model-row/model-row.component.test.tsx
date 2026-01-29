// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { getMockedModel } from '../../../../../../mocks/mock-model';
import type { Model } from '../../../../../constants/shared-types';
import { ModelRow } from './model-row.component';

describe('ModelRow', () => {
    const defaultModel: Model = getMockedModel({
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

    describe('basic rendering', () => {
        it('should render all model information correctly', () => {
            render(<ModelRow model={defaultModel} />);

            expect(screen.getByTestId('model-name')).toHaveTextContent('Test Model');
            expect(screen.getByText(/YOLOX \(Apache 2\.0\)/)).toBeInTheDocument();
            expect(screen.getByText('Speed')).toBeInTheDocument();
        });

        it('should render "Unnamed Model" when model name is null or undefined', () => {
            const modelWithoutName = getMockedModel({ name: undefined });

            render(<ModelRow model={modelWithoutName} />);

            expect(screen.getByTestId('model-name')).toHaveTextContent('Unnamed Model');
        });

        it('should render "-" when model size is 0 or negative', () => {
            const modelWithZeroSize = getMockedModel({ size: 0 });

            render(<ModelRow model={modelWithZeroSize} />);

            expect(screen.getByText('-')).toBeInTheDocument();
        });
    });

    describe('active model tag', () => {
        it('should show active tag only when model id matches activeModelArchitectureId', () => {
            const { rerender } = render(<ModelRow model={defaultModel} activeModelArchitectureId='model-123' />);
            expect(screen.getByText('Active')).toBeInTheDocument();

            rerender(<ModelRow model={defaultModel} activeModelArchitectureId={'different-id'} />);
            expect(screen.queryByText('Active')).not.toBeInTheDocument();

            rerender(<ModelRow model={defaultModel} />);
            expect(screen.queryByText('Active')).not.toBeInTheDocument();
        });
    });

    describe('parent revision model', () => {
        it('should render parent revision model when provided and call onExpandModel when clicked', async () => {
            const onExpandModel = vi.fn();
            const parentModel: Model = getMockedModel({
                id: 'parent-123',
                name: 'Parent Model',
            });

            render(<ModelRow model={defaultModel} parentRevisionModel={parentModel} onExpandModel={onExpandModel} />);

            expect(screen.getByText('Fine-tuned from')).toBeInTheDocument();
            const parentLink = screen.getByRole('link', { name: 'Parent Model' });
            expect(parentLink).toBeInTheDocument();

            await userEvent.click(parentLink);
            expect(onExpandModel).toHaveBeenCalledWith('parent-123');
        });

        it('should not render parent revision model when not provided', () => {
            render(<ModelRow model={defaultModel} />);

            expect(screen.queryByText('Fine-tuned from')).not.toBeInTheDocument();
        });
    });

    describe('model actions menu', () => {
        it('should render menu with all items and handle clicks correctly', async () => {
            const onModelAction = vi.fn();

            render(<ModelRow model={defaultModel} onModelAction={onModelAction} />);

            const menuButton = screen.getByRole('button', { name: 'Model actions' });
            expect(menuButton).toBeInTheDocument();

            await userEvent.click(menuButton);
            expect(screen.getByRole('menuitem', { name: 'Set as active' })).toBeInTheDocument();
            expect(screen.getByRole('menuitem', { name: 'Rename' })).toBeInTheDocument();
            expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();

            await userEvent.click(screen.getByRole('menuitem', { name: 'Set as active' }));
            expect(onModelAction).toHaveBeenCalledWith('active');
            await userEvent.click(menuButton);
            await userEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));
            expect(onModelAction).toHaveBeenCalledWith('rename');

            await userEvent.click(menuButton);
            await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));
            expect(onModelAction).toHaveBeenCalledWith('delete');

            expect(onModelAction).toHaveBeenCalledTimes(3);
        });

        it('should not render actions menu when onModelAction is not provided', () => {
            render(<ModelRow model={defaultModel} />);

            expect(screen.queryByRole('button', { name: 'Model actions' })).not.toBeInTheDocument();
        });
    });
});
