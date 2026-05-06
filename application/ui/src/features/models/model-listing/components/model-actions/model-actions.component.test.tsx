// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModel } from 'mocks/mock-model';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { downloadFile } from '../../../../../shared/util';
import { ModelActions } from './model-actions.component';

vi.mock('../../../../../shared/util', async (importActual) => {
    const actual = await importActual<typeof import('../../../../../shared/util')>();

    return {
        ...actual,
        downloadFile: vi.fn(),
    };
});

const mockModel = getMockedModel({
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

describe('ModelActions', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should render menu with all items', async () => {
        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        expect(menuButton).toBeInTheDocument();

        await userEvent.click(menuButton);

        expect(screen.getByRole('menuitem', { name: 'Set as active' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Rename' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete weights' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Delete model' })).toBeInTheDocument();
    });

    it('should open rename dialog when rename action is clicked', async () => {
        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));

        expect(screen.getByRole('dialog', { name: 'Rename model' })).toBeInTheDocument();
        expect(screen.getByRole('textbox', { name: /Model name/ })).toHaveValue('Test Model');
    });

    it('should open delete weights dialog when delete weights action is clicked', async () => {
        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Delete weights' }));

        expect(screen.getByRole('alertdialog', { name: 'Delete weights' })).toBeInTheDocument();
        expect(
            screen.getByText(`Are you sure you want to delete the weights for model "${mockModel.name}"?`)
        ).toBeInTheDocument();
    });

    it('should open delete dialog when delete action is clicked', async () => {
        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Delete model' }));

        expect(screen.getByRole('alertdialog', { name: 'Delete model' })).toBeInTheDocument();
        expect(screen.getByText(/This action cannot be undone/)).toBeInTheDocument();
    });

    it('should show download logs action in training logs dialog', async () => {
        server.use(
            http.get('/api/projects/{project_id}/models/{model_id}/logs', () => {
                return new HttpResponse(
                    new Blob(
                        ['[2025-01-10 10:00:00][INFO ] Initializing\n[2025-01-10 10:00:01][INFO ] Training started'],
                        {
                            type: 'text/plain',
                        }
                    ),
                    {
                        status: 200,
                        headers: {
                            'content-type': 'text/plain',
                        },
                    }
                );
            })
        );

        render(<ModelActions model={mockModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'View training logs' }));

        await userEvent.click(await screen.findByRole('button', { name: 'Download logs' }));

        expect(downloadFile).toHaveBeenCalledWith(expect.stringMatching(/^blob:/), `training-logs-${mockModel.id}.log`);
        expect(await screen.findByText('Training logs downloaded successfully')).toBeInTheDocument();
    });

    it('should disable "Set as active" and "Rename" when model is currently training', async () => {
        const trainingModel = getMockedModel({
            ...mockModel,
            training_info: {
                status: 'in_progress',
                label_schema_revision: {
                    labels: [],
                },
                start_time: '2025-01-10T10:00:00.000000+00:00',
                end_time: null,
                dataset_revision_id: 'dataset-123',
            },
        });

        render(<ModelActions model={trainingModel} />);

        const menuButton = screen.getByRole('button', { name: 'Model actions' });
        await userEvent.click(menuButton);

        const setActiveItem = screen.getByRole('menuitem', { name: 'Set as active' });
        const renameItem = screen.getByRole('menuitem', { name: 'Rename' });
        const deleteItem = screen.getByRole('menuitem', { name: 'Delete model' });

        expect(setActiveItem).toHaveAttribute('aria-disabled', 'true');
        expect(renameItem).toHaveAttribute('aria-disabled', 'true');
        expect(deleteItem).not.toHaveAttribute('aria-disabled', 'true');
    });

    describe('when model has deleted weights', () => {
        const modelWithDeletedWeights = getMockedModel({
            ...mockModel,
            files_deleted: true,
        });

        it('disables "Delete weights", "View training logs", and "Set as active"', async () => {
            render(<ModelActions model={modelWithDeletedWeights} />);

            const menuButton = screen.getByRole('button', { name: 'Model actions' });
            await userEvent.click(menuButton);

            expect(screen.getByRole('menuitem', { name: 'Delete weights' })).toHaveAttribute('aria-disabled', 'true');
            expect(screen.getByRole('menuitem', { name: 'View training logs' })).toHaveAttribute(
                'aria-disabled',
                'true'
            );
            expect(screen.getByRole('menuitem', { name: 'Set as active' })).toHaveAttribute('aria-disabled', 'true');
        });

        it('keeps "Rename" and "Delete model" enabled', async () => {
            render(<ModelActions model={modelWithDeletedWeights} />);

            const menuButton = screen.getByRole('button', { name: 'Model actions' });
            await userEvent.click(menuButton);

            expect(screen.getByRole('menuitem', { name: 'Rename' })).not.toHaveAttribute('aria-disabled', 'true');
            expect(screen.getByRole('menuitem', { name: 'Delete model' })).not.toHaveAttribute('aria-disabled', 'true');
        });
    });
});
