// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ImportDatasetAsNewProjectState } from '../../../../dataset/import-export/import-dataset/util';
import { ProgressStepper } from './progress-stepper.component';

describe('ProgressStepper', () => {
    it.each(['uploading', 'preparing'])('marks first step as active for %s', (currentStep) => {
        render(<ProgressStepper currentStep={currentStep as ImportDatasetAsNewProjectState} />);

        const stepOne = screen.getByLabelText('step one');
        const stepTwo = screen.getByLabelText('step two');
        const stepThree = screen.getByLabelText('step three');

        expect(stepOne).toHaveAttribute('data-active', 'true');
        expect(stepTwo).toHaveAttribute('data-active', 'false');
        expect(stepThree).toHaveAttribute('data-active', 'false');
    });

    it('shows first step as completed and second as active for task type selection', () => {
        render(<ProgressStepper currentStep={'taskTypeSelection' as const} />);

        const stepOne = screen.getByLabelText('step one');
        const stepTwo = screen.getByLabelText('step two');
        const stepThree = screen.getByLabelText('step three');

        expect(stepOne).toHaveTextContent('✓');
        expect(stepTwo).toHaveTextContent('2');
        expect(stepThree).toHaveTextContent('3');

        expect(stepOne).toHaveAttribute('data-active', 'false');
        expect(stepTwo).toHaveAttribute('data-active', 'true');
        expect(stepThree).toHaveAttribute('data-active', 'false');
    });

    it('shows first two steps as completed and third as active for label mapping', () => {
        render(<ProgressStepper currentStep={'labelMapping' as const} />);

        const stepOne = screen.getByLabelText('step one');
        const stepTwo = screen.getByLabelText('step two');
        const stepThree = screen.getByLabelText('step three');

        expect(stepOne).toHaveTextContent('✓');
        expect(stepTwo).toHaveTextContent('✓');
        expect(stepThree).toHaveTextContent('3');

        expect(stepOne).toHaveAttribute('data-active', 'false');
        expect(stepTwo).toHaveAttribute('data-active', 'false');
        expect(stepThree).toHaveAttribute('data-active', 'true');
    });

    it('renders static step labels', () => {
        render(<ProgressStepper currentStep={'uploading' as const} />);

        expect(screen.getByText('Dataset')).toBeVisible();
        expect(screen.getByText('Task type')).toBeVisible();
        expect(screen.getByText('Label')).toBeVisible();
    });
});
