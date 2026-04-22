// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { Flex, StatusLight, View } from '@geti/ui';

import { type AnnotatorMode } from '../../../../../shared/annotator/annotator-mode';

import classes from './annotator-modes.module.scss';

interface ToggleButtonProps {
    children: ReactNode;
    isActive: boolean;
    onClick: () => void;
}

const ToggleButton = ({ children, isActive, onClick }: ToggleButtonProps) => {
    return (
        <button aria-pressed={isActive} onClick={onClick} className={classes.toggleButton}>
            {children}
        </button>
    );
};

interface ToggleButtonWithCueProps extends ToggleButtonProps {
    showCue: boolean;
    cueLabel: string;
}

const ToggleButtonWithCue = ({ showCue, cueLabel, onClick, isActive, children }: ToggleButtonWithCueProps) => {
    return (
        <span className={classes.buttonWrapper}>
            <ToggleButton isActive={isActive} onClick={onClick}>
                {children}
            </ToggleButton>
            {showCue && (
                <StatusLight
                    variant={'info'}
                    role={'status'}
                    aria-label={cueLabel}
                    UNSAFE_className={classes.availabilityCue}
                />
            )}
        </span>
    );
};

interface AnnotatorModesProps {
    mode: AnnotatorMode;
    onModeChange: (mode: AnnotatorMode) => void;
    hasAnnotations: boolean;
    hasPredictions: boolean;
}

export const AnnotatorModes = ({ mode, onModeChange, hasAnnotations, hasPredictions }: AnnotatorModesProps) => {
    const [dismissedCues, setDismissedCues] = useState<Set<AnnotatorMode>>(new Set());

    const handleModeChange = (nextMode: AnnotatorMode) => {
        const hasContent = nextMode === 'annotation' ? hasAnnotations : hasPredictions;

        if (hasContent) {
            setDismissedCues((prev) => new Set(prev).add(nextMode));
        }

        onModeChange(nextMode);
    };

    return (
        <View backgroundColor={'gray-200'} padding={'size-50'} borderRadius={'regular'}>
            <Flex
                width={'100%'}
                height={'100%'}
                gap={'size-100'}
                alignItems={'center'}
                data-testid={'annotator-modes-id'}
            >
                <ToggleButtonWithCue
                    isActive={mode === 'annotation'}
                    onClick={() => handleModeChange('annotation')}
                    showCue={mode === 'prediction' && hasAnnotations && !dismissedCues.has('annotation')}
                    cueLabel={'Annotation available'}
                >
                    Annotation
                </ToggleButtonWithCue>
                <ToggleButtonWithCue
                    isActive={mode === 'prediction'}
                    onClick={() => handleModeChange('prediction')}
                    showCue={
                        mode === 'annotation' && !hasAnnotations && hasPredictions && !dismissedCues.has('prediction')
                    }
                    cueLabel={'Prediction available'}
                >
                    Prediction
                </ToggleButtonWithCue>
            </Flex>
        </View>
    );
};
