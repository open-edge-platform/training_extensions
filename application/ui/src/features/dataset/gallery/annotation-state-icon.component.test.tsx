// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render } from '@testing-library/react';

import type { MediaItemState } from '../../../constants/shared-types';
import { AnnotationStatusIcon } from './annotation-state-icon.component';

describe('AnnotationStatusIcon', () => {
    it('renders correct icon for each state', () => {
        const states: Array<MediaItemState | undefined> = ['accepted', 'rejected', undefined];

        states.forEach((state) => {
            const { container } = render(<AnnotationStatusIcon state={state} />);
            expect(container.querySelector('svg')).toBeInTheDocument();
        });

        // Verify different icons are rendered for different states
        const { container: acceptedContainer } = render(<AnnotationStatusIcon state='accepted' />);
        const { container: rejectedContainer } = render(<AnnotationStatusIcon state='rejected' />);
        const { container: reviewContainer } = render(<AnnotationStatusIcon state={undefined} />);

        expect(acceptedContainer.querySelector('svg')).toBeInTheDocument();
        expect(rejectedContainer.querySelector('svg')).toBeInTheDocument();
        expect(reviewContainer.querySelector('svg')).toBeInTheDocument();
    });
});
