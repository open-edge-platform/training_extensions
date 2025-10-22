// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Text, View } from '@geti/ui';

const DEPRECATED_MODEL_EXPLANATION =
    `This architecture is no longer actively developed. ` +
    `In the future you will no longer be able to train new model versions on this architecture.`;
const OBSOLETE_MODEL_EXPLANATION =
    'The architecture is no longer supported, which means it cannot be used for training new models, ' +
    'but previously trained models can be still evaluated, optimized, and deployed.';

interface ModelArchitectureTooltipTextProps {
    isDeprecated?: boolean;
    isObsolete?: boolean;
    description: string;
}

export const ModelArchitectureTooltipText = ({
    isDeprecated,
    isObsolete,
    description,
}: ModelArchitectureTooltipTextProps) => {
    return (
        <View>
            <Text>{description}</Text>
            {isDeprecated && (
                <>
                    <Divider marginY={'size-150'} height={'size-10'} />
                    <Text>Deprecated model tag</Text>
                    <Text marginTop={'size-50'} UNSAFE_style={{ display: 'block' }}>
                        {DEPRECATED_MODEL_EXPLANATION}
                    </Text>
                </>
            )}
            {isObsolete && (
                <>
                    <Divider marginY={'size-150'} height={'size-10'} />
                    <Text>Obsolete model tag</Text>
                    <Text marginTop={'size-50'} UNSAFE_style={{ display: 'block' }}>
                        {OBSOLETE_MODEL_EXPLANATION}
                    </Text>
                </>
            )}
        </View>
    );
};
