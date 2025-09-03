// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';

import { ModelSelectionGroup } from './model-selection-group.component';

describe('ModelSelectionGroup', () => {
    it('renders all model options', () => {
        render(<ModelSelectionGroup />);

        expect(screen.getByLabelText('Model option: Object Detection')).toBeInTheDocument();
        expect(screen.getByLabelText('Model option: Image Segmentation')).toBeInTheDocument();
        expect(screen.getByLabelText('Model option: Image Classification')).toBeInTheDocument();
    });

    it('selects the first model by default', () => {
        render(<ModelSelectionGroup />);

        const radio = screen.getByLabelText('detection');
        expect(radio).toBeChecked();
    });

    it('selects a model when the whole element is clicked', () => {
        render(<ModelSelectionGroup />);

        const segOption = screen.getByLabelText('Model option: Image Segmentation');
        fireEvent.click(segOption);

        const segRadio = screen.getByLabelText('segmentation');
        expect(segRadio).toBeChecked();
    });

    it('selects a model when the radio element is clicked', () => {
        render(<ModelSelectionGroup />);

        const classRadio = screen.getByLabelText('classification');
        fireEvent.click(classRadio);

        expect(classRadio).toBeChecked();
    });

    it('only one model is selected at a time', () => {
        render(<ModelSelectionGroup />);

        const segOption = screen.getByLabelText('Model option: Image Segmentation');
        fireEvent.click(segOption);

        expect(screen.getByLabelText('segmentation')).toBeChecked();
        expect(screen.getByLabelText('detection')).not.toBeChecked();
        expect(screen.getByLabelText('classification')).not.toBeChecked();
    });
});
