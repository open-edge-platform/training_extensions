// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text, ToggleButtons } from '@geti/ui';

import { Tooltip } from '../../ui/tooltip.component';

import styles from './tiling.module.scss';

export enum TILING_MODES {
    OFF = 'Off',
    AUTOMATIC = 'Automatic',
    CUSTOM = 'Custom',
}

export const TilingModeTooltip = () => {
    return (
        <Tooltip>
            Tiling is a technique that divides high-resolution images into smaller tiles and might be useful to increase
            accuracy for small object detection tasks.
        </Tooltip>
    );
};

type TilingModesProps = {
    selectedTilingMode: TILING_MODES;
    onTilingModeChange: (tilingMode: TILING_MODES) => void;
};

export const TilingModes = ({ selectedTilingMode, onTilingModeChange }: TilingModesProps) => {
    return (
        <>
            <Text UNSAFE_className={styles.title}>
                Tiling mode <TilingModeTooltip />
            </Text>
            <ToggleButtons
                options={[TILING_MODES.OFF, TILING_MODES.AUTOMATIC, TILING_MODES.CUSTOM]}
                selectedOption={selectedTilingMode}
                onOptionChange={onTilingModeChange}
            />
        </>
    );
};
