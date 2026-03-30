// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, View } from '@geti-ui/ui';

import { Label } from '../../../../../../constants/shared-types';

import classes from './labels.module.scss';

type LabelsProps = {
    labels: Label[];
};

export const Labels = ({ labels }: LabelsProps) => {
    return (
        <Flex direction={'column'} gap={'size-100'}>
            {labels.map((label) => (
                <View
                    key={label.id}
                    height={'size-225'}
                    paddingX={'size-50'}
                    backgroundColor={'gray-200'}
                    UNSAFE_className={classes.label}
                >
                    <span title={label.name}>{label.name}</span>
                </View>
            ))}
        </Flex>
    );
};
