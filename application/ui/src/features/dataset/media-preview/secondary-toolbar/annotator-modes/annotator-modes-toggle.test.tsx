// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { AnnotatorMode } from '../../../../../shared/annotator/annotator-mode';
import { AnnotatorModes } from './annotator-modes-toggle.component';

const ControlledAnnotatorModes = ({
    initialMode,
    hasAnnotations,
    hasPredictions,
}: {
    initialMode: AnnotatorMode;
    hasAnnotations: boolean;
    hasPredictions: boolean;
}) => {
    const [mode, setMode] = useState<AnnotatorMode>(initialMode);

    return (
        <AnnotatorModes
            mode={mode}
            onModeChange={setMode}
            hasAnnotations={hasAnnotations}
            hasPredictions={hasPredictions}
        />
    );
};

const renderComponent = ({
    mode = 'annotation',
    hasAnnotations = false,
    hasPredictions = false,
}: {
    mode: AnnotatorMode;
    hasAnnotations?: boolean;
    hasPredictions?: boolean;
}) => {
    return render(
        <ControlledAnnotatorModes initialMode={mode} hasAnnotations={hasAnnotations} hasPredictions={hasPredictions} />
    );
};

describe('AnnotatorModes', () => {
    describe('Button active state', () => {
        it('marks annotation button as active when mode is annotation', () => {
            renderComponent({ mode: 'annotation' });

            expect(screen.getByRole('button', { name: 'Annotation' })).toHaveAttribute('aria-pressed', 'true');
            expect(screen.getByRole('button', { name: 'Prediction' })).toHaveAttribute('aria-pressed', 'false');
        });

        it('marks prediction button as active when mode is prediction', () => {
            renderComponent({ mode: 'prediction' });

            expect(screen.getByRole('button', { name: 'Prediction' })).toHaveAttribute('aria-pressed', 'true');
            expect(screen.getByRole('button', { name: 'Annotation' })).toHaveAttribute('aria-pressed', 'false');
        });

        it('marks button as active when clicked', () => {
            renderComponent({ mode: 'prediction' });

            fireEvent.click(screen.getByRole('button', { name: 'Annotation' }));

            expect(screen.getByRole('button', { name: 'Annotation' })).toHaveAttribute('aria-pressed', 'true');
            expect(screen.getByRole('button', { name: 'Prediction' })).toHaveAttribute('aria-pressed', 'false');

            fireEvent.click(screen.getByRole('button', { name: 'Prediction' }));

            expect(screen.getByRole('button', { name: 'Prediction' })).toHaveAttribute('aria-pressed', 'true');
            expect(screen.getByRole('button', { name: 'Annotation' })).toHaveAttribute('aria-pressed', 'false');
        });
    });

    describe('Prediction cue visibility', () => {
        it('shows prediction cue when mode is annotation, has predictions, and has no annotations', () => {
            renderComponent({ mode: 'annotation', hasPredictions: true, hasAnnotations: false });

            expect(screen.getByRole('status', { name: 'Prediction available' })).toBeInTheDocument();
        });

        it('does not show prediction cue when has annotations', () => {
            renderComponent({ mode: 'annotation', hasPredictions: true, hasAnnotations: true });

            expect(screen.queryByRole('status', { name: 'Prediction available' })).not.toBeInTheDocument();
        });

        it('does not show prediction cue when mode is prediction', () => {
            renderComponent({ mode: 'prediction', hasPredictions: true, hasAnnotations: false });

            expect(screen.queryByRole('status', { name: 'Prediction available' })).not.toBeInTheDocument();
        });
    });

    describe('Annotation cue visibility', () => {
        it('shows annotation cue when mode is prediction and has annotations', () => {
            renderComponent({ mode: 'prediction', hasAnnotations: true });

            expect(screen.getByRole('status', { name: 'Annotation available' })).toBeInTheDocument();
        });

        it('does not show annotation cue when mode is annotation', () => {
            renderComponent({ mode: 'annotation', hasAnnotations: true });

            expect(screen.queryByRole('status', { name: 'Annotation available' })).not.toBeInTheDocument();
        });
    });

    describe('Cue dismissal', () => {
        it('dismisses annotation cue after clicking annotation button when has annotation', () => {
            renderComponent({ mode: 'prediction', hasPredictions: false, hasAnnotations: true });

            expect(screen.getByRole('status', { name: 'Annotation available' })).toBeInTheDocument();

            fireEvent.click(screen.getByRole('button', { name: 'Annotation' }));

            fireEvent.click(screen.getByRole('button', { name: 'Prediction' }));

            expect(screen.queryByRole('status', { name: 'Annotation available' })).not.toBeInTheDocument();
        });

        it('dismisses prediction cue after clicking prediction button when has predictions', () => {
            renderComponent({ mode: 'annotation', hasPredictions: true, hasAnnotations: false });

            expect(screen.getByRole('status', { name: 'Prediction available' })).toBeInTheDocument();

            fireEvent.click(screen.getByRole('button', { name: 'Prediction' }));

            fireEvent.click(screen.getByRole('button', { name: 'Annotation' }));

            expect(screen.queryByRole('status', { name: 'Prediction available' })).not.toBeInTheDocument();
        });
    });
});
