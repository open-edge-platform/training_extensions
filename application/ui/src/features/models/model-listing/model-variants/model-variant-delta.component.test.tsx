// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ModelVariantDelta } from './model-variant-delta.component';

describe('ModelVariantDelta', () => {
    it('shows a positive change in green by default', () => {
        render(<ModelVariantDelta currentValue={110} baselineValue={100} />);

        const delta = screen.getByTestId('model-variant-delta-accuracy');

        expect(delta).toHaveTextContent('+10%');
        expect(delta).toHaveStyle({ color: 'var(--moss-tint-1)' });
    });

    it('treats negative size change as positive', () => {
        render(<ModelVariantDelta currentValue={80} baselineValue={100} changeType='size' />);

        const delta = screen.getByTestId('model-variant-delta-size');

        expect(delta).toHaveTextContent('-20%');
        expect(delta).toHaveStyle({ color: 'var(--moss-tint-1)' });
    });

    it('treats negative accuracy change as negative', () => {
        render(<ModelVariantDelta currentValue={80} baselineValue={100} changeType='accuracy' />);

        const delta = screen.getByTestId('model-variant-delta-accuracy');

        expect(delta).toHaveTextContent('-20%');
        expect(delta).toHaveStyle({ color: 'var(--coral-shade-1)' });
    });

    it('renders nothing when values are equal', () => {
        render(<ModelVariantDelta currentValue={100} baselineValue={100} />);

        expect(screen.queryByTestId('model-variant-delta-accuracy')).not.toBeInTheDocument();
    });
});
