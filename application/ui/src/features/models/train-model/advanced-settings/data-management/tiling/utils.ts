// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ConfigurationParameter } from '../../../../configuration.interface';
import { isBoolParameter } from '../../utils';
import { TILING_MODES } from './tiling-modes.component';

export const TILING_OFF_DESCRIPTION =
    'Model processes the entire image as a single unit without dividing it into smaller tiles. This approach ' +
    'is straightforward but may struggle with detecting small objects in high-resolution images, as the model ' +
    'might miss finer details';

export const TILING_AUTOMATIC_DESCRIPTION =
    'It means that the system will automatically set the parameters based on the images resolution and ' +
    'annotations size.';

const ADAPTIVE_TILING_PARAMETER = 'adaptive_tiling';
const ENABLE_TILING_PARAMETER = 'enable';

export const getAdaptiveTilingParameter = (tilingParameters: ConfigurationParameter[]) => {
    const parameter = tilingParameters.find(({ key }) => key === ADAPTIVE_TILING_PARAMETER);

    if (parameter === undefined || !isBoolParameter(parameter)) {
        return undefined;
    }

    return parameter;
};

export const getEnableTilingParameter = (tilingParameters: ConfigurationParameter[]) => {
    const parameter = tilingParameters.find(({ key }) => key === ENABLE_TILING_PARAMETER);

    if (parameter === undefined || !isBoolParameter(parameter)) {
        return undefined;
    }

    return parameter;
};

export const getTilingMode = (tilingParameters: ConfigurationParameter[]): TILING_MODES => {
    const adaptive = getAdaptiveTilingParameter(tilingParameters);
    const enablingTiling = getEnableTilingParameter(tilingParameters);

    if (enablingTiling?.value === false) {
        return TILING_MODES.OFF;
    }

    if (adaptive?.value === true) {
        return TILING_MODES.AUTOMATIC;
    }

    return TILING_MODES.CUSTOM;
};

export const getCustomTilingParameters = (parameters: ConfigurationParameter[]) => {
    return parameters.filter(
        (parameter) => ![ADAPTIVE_TILING_PARAMETER, ENABLE_TILING_PARAMETER].includes(parameter.key)
    );
};
