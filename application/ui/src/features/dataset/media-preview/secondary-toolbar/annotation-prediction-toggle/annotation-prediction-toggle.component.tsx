// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { Flex, View } from '@geti/ui';

import styles from './annotation-prediction-toggle.module.scss';

type Mode = 'prediction' | 'annotation';

type ToggleButtonProps = {
    children: ReactNode;
    selectedMode: Mode;
    mode: Mode;
    onClick: () => void;
};

const ToggleButton = ({ children, mode, selectedMode, onClick }: ToggleButtonProps) => {
    return (
        <button aria-pressed={mode === selectedMode} onClick={onClick} className={styles.toggleButton}>
            {children}
        </button>
    );
};

export const AnnotationPredictionToggle = () => {
    const [mode, setMode] = useState<Mode>('annotation');

    return (
        <View backgroundColor={'gray-200'} padding={'size-50'} borderRadius={'regular'}>
            <Flex width={'100%'} height={'100%'} gap={'size-100'} alignItems={'center'}>
                <ToggleButton mode={'annotation'} selectedMode={mode} onClick={() => setMode('annotation')}>
                    Annotation
                </ToggleButton>
                <ToggleButton mode={'prediction'} selectedMode={mode} onClick={() => setMode('prediction')}>
                    Prediction
                </ToggleButton>
            </Flex>
        </View>
    );
};
