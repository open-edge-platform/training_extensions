// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { dimensionValue, Divider, Flex, Text, View } from '@geti/ui';

import { isNonEmptyString } from '../../shared/util';

type JobStatusCardProps = {
    title: string;
    message?: string;
    bottomIcon?: ReactNode;
    bottomLeftMessage: ReactNode;
    bottomRightMessage?: string;
    actionButtons: ReactNode;
};

export const JobStatusCard = ({
    title,
    message,
    actionButtons,
    bottomLeftMessage,
    bottomRightMessage,
    bottomIcon = null,
}: JobStatusCardProps) => {
    return (
        <View
            padding={'size-150'}
            position={'relative'}
            borderWidth={'thin'}
            borderColor={'gray-200'}
            borderRadius={'regular'}
            backgroundColor={'gray-75'}
        >
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>{title}</Text>

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    {actionButtons}
                </Flex>
            </Flex>

            {isNonEmptyString(message) && <Text>{message}</Text>}

            <Divider size='S' marginY='size-150' />

            <Flex justifyContent='space-between'>
                <Flex gap={'size-100'} direction={'column'}>
                    {bottomIcon}

                    {isNonEmptyString(bottomLeftMessage) ? <Text>{bottomLeftMessage}</Text> : bottomLeftMessage}
                </Flex>

                {isNonEmptyString(bottomRightMessage) && <Text>{bottomRightMessage}</Text>}
            </Flex>
        </View>
    );
};
