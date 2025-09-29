// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@test-utils/render';
import userEvent from '@testing-library/user-event';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { WebcamSourceConfig } from '../util';
import { Webcam } from './webcam.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('../hooks/use-source-action.hook');

const mockedConfig: WebcamSourceConfig = {
    id: '1',
    name: 'Test Folder',
    source_type: 'webcam',
    device_id: 0,
};

describe('Webcam', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(<Webcam />);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<Webcam />);

        userEvent.click(screen.getByRole('button', { name: 'Apply' }));

        waitFor(() => {
            expect(mockedSubmitAction).toHaveBeenCalled();
        });
    });
});
