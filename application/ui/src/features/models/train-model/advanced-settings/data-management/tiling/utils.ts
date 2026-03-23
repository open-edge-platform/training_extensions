// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ConfigurableParameter,
    ConfigurableParameterGroup,
    TrainingConfiguration,
} from '../../../../../../constants/shared-types';
import { findGroupByKey, isParameter } from '../../../../model-listing/model-training-parameters/utils';
import { isBoolParameter } from '../../utils';

export type TilingConfigurableParameterGroup = Omit<ConfigurableParameterGroup, 'parameters'> & {
    parameters: ConfigurableParameter[];
};

export const getTilingParameters = (
    trainingConfiguration: TrainingConfiguration
): TilingConfigurableParameterGroup | undefined => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;
    const dataAugmentation = findGroupByKey(datasetPreparation, 'augmentation')?.parameters;
    const tilingParameters = findGroupByKey(dataAugmentation, 'tiling');

    if (tilingParameters === undefined || tilingParameters.parameters === undefined) return undefined;

    return {
        ...tilingParameters,
        parameters: tilingParameters.parameters.filter(isParameter),
    };
};

export const TILING_OFF_DESCRIPTION =
    'Model processes the entire image as a single unit without dividing it into smaller tiles. This approach ' +
    'is straightforward but may struggle with detecting small objects in high-resolution images, as the model ' +
    'might miss finer details';

export const TILING_AUTOMATIC_DESCRIPTION =
    'It means that the system will automatically set the parameters based on the image resolution and ' +
    'annotations size.';

const ADAPTIVE_TILING_PARAMETER = 'enable_adaptive_tiling';
const ENABLE_TILING_PARAMETER = 'enable';

const getBoolParameter = (tilingParameters: ConfigurableParameter[], key: string) => {
    const parameter = tilingParameters.find((tilingParameter) => key === tilingParameter.key);

    if (parameter === undefined || !isBoolParameter(parameter)) {
        return undefined;
    }

    return parameter;
};

export const getAdaptiveTilingParameter = (tilingParameters: ConfigurableParameter[]) => {
    return getBoolParameter(tilingParameters, ADAPTIVE_TILING_PARAMETER);
};

export const getEnableTilingParameter = (tilingParameters: ConfigurableParameter[]) => {
    return getBoolParameter(tilingParameters, ENABLE_TILING_PARAMETER);
};

export const TILING_MODES = {
    OFF: 'Off',
    AUTOMATIC: 'Automatic',
    CUSTOM: 'Custom',
} as const;

export type TilingMode = (typeof TILING_MODES)[keyof typeof TILING_MODES];

export const getTilingMode = (tilingParameters: ConfigurableParameter[]): TilingMode => {
    const adaptive = getAdaptiveTilingParameter(tilingParameters);
    const enablingTiling = getEnableTilingParameter(tilingParameters);

    if (!enablingTiling || enablingTiling.value === false) {
        return TILING_MODES.OFF;
    }

    if (adaptive?.value === true) {
        return TILING_MODES.AUTOMATIC;
    }

    return TILING_MODES.CUSTOM;
};

export const getCustomTilingParameters = (parameters: ConfigurableParameter[]) => {
    return parameters.filter(
        (parameter) => ![ADAPTIVE_TILING_PARAMETER, ENABLE_TILING_PARAMETER].includes(parameter.key)
    );
};
