// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { DatasetCard } from './dataset-card.component';

describe('DatasetCard', () => {
    it('does not render the expand button when hasFullSizeContent is false', () => {
        render(
            <DatasetCard title='Test Title' gridArea='card'>
                <div>Card content</div>
            </DatasetCard>
        );

        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });

    it('renders the expand button when hasFullSizeContent is true', () => {
        render(
            <DatasetCard title='Test Title' gridArea='card' hasFullSizeContent>
                <div>Card content</div>
            </DatasetCard>
        );

        expect(screen.getByText('Test Title')).toBeVisible();
        expect(screen.getByText('Card content')).toBeVisible();
        expect(screen.getByRole('button')).toBeVisible();
    });

    it('opens fullscreen dialog when expand is clicked', async () => {
        render(
            <DatasetCard title='Test Title' gridArea='card' hasFullSizeContent>
                <div>Card content</div>
            </DatasetCard>
        );

        expect(screen.getByText('Card content')).toBeVisible();

        await userEvent.click(screen.getByRole('button'));

        expect(await screen.findByRole('button', { name: 'collapse fullscreen' })).toBeVisible();
    });
});
