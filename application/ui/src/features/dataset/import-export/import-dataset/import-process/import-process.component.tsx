// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Text } from '@geti/ui';

import { CircularProgress } from '../../../../../components/circular-progress/circular-progress.component';

import classes from './import-process.module.scss';

export const ImportProcess = () => {
    return (
        <Flex
            width='100%'
            height='100%'
            gap='size-275'
            direction='column'
            alignItems='center'
            justifyContent='center'
            UNSAFE_style={{ padding: dimensionValue('size-500') }}
        >
            <CircularProgress
                size={80}
                percentage={10}
                strokeWidth={8}
                labelFontSize={12}
                color='static-blue-200'
                labelFontColor='gray-700'
                backStrokeColor='gray-75'
            />

            <Flex direction='column' alignItems='center' justifyContent='center'>
                <Text UNSAFE_className={classes.title}>Uploading</Text>
                <Text UNSAFE_className={classes.description}>Dataset is being uploaded</Text>

                <Text marginTop='size-100'>File name...</Text>
            </Flex>
        </Flex>
    );
};
