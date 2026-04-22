// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Flex, View } from '@geti/ui';

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
    return (
        <View backgroundColor={'gray-200'} padding={'size-50'} borderRadius={'regular'}>
            <Flex
                width={'100%'}
                height={'100%'}
                gap={'size-100'}
                alignItems={'center'}
                data-testid={'annotator-modes-id'}
            >
                <ToggleButton mode={'annotation'} selectedMode={mode} onClick={() => onModeChange('annotation')}>
                    Annotation
                </ToggleButton>
                <ToggleButton mode={'prediction'} selectedMode={mode} onClick={() => onModeChange('prediction')}>
                    Prediction
                </ToggleButton>
            </Flex>
        </View>
    );
};
