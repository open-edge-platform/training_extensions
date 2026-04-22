// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Flex, StatusLight, View } from '@geti/ui';

import { type AnnotatorMode } from '../../../../../shared/annotator/annotator-mode';

import styles from './annotator-modes.module.scss';

type ToggleButtonProps = {
    children: ReactNode;
    selectedMode: AnnotatorMode;
    mode: AnnotatorMode;
    onClick: () => void;
};

const ToggleButton = ({ children, mode, selectedMode, onClick }: ToggleButtonProps) => {
    return (
        <button aria-pressed={mode === selectedMode} onClick={onClick} className={styles.toggleButton}>
            {children}
        </button>
    );
};

type AnnotatorModes = {
    mode: AnnotatorMode;
    onModeChange: (mode: AnnotatorMode) => void;
    hasAnnotations: boolean;
    hasPredictions: boolean;
};

export const AnnotatorModes = ({ mode, onModeChange, hasAnnotations, hasPredictions }: AnnotatorModes) => {
    const showAnnotationCue = mode === 'prediction' && hasAnnotations;
    const showPredictionCue = mode === 'annotation' && hasPredictions;

    return (
        <View backgroundColor={'gray-200'} padding={'size-50'} borderRadius={'regular'}>
            <Flex
                width={'100%'}
                height={'100%'}
                gap={'size-100'}
                alignItems={'center'}
                data-testid={'annotator-modes-id'}
            >
                <span className={styles.buttonWrapper}>
                    <ToggleButton mode={'annotation'} selectedMode={mode} onClick={() => onModeChange('annotation')}>
                        Annotation
                    </ToggleButton>
                    {showAnnotationCue && (
                        <StatusLight
                            variant={'info'}
                            role={'status'}
                            aria-label={'Annotation available'}
                            UNSAFE_className={styles.availabilityCue}
                        />
                    )}
                </span>
                <span className={styles.buttonWrapper}>
                    <ToggleButton mode={'prediction'} selectedMode={mode} onClick={() => onModeChange('prediction')}>
                        Prediction
                    </ToggleButton>
                    {showPredictionCue && (
                        <StatusLight
                            variant={'info'}
                            role={'status'}
                            aria-label={'Prediction available'}
                            UNSAFE_className={styles.availabilityCue}
                        />
                    )}
                </span>
            </Flex>
        </View>
    );
};
