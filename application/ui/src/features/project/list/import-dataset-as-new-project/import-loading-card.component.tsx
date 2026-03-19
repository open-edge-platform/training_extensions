// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Loading, View } from '@geti/ui';

export const ImportLoadingCard = () => {
    return (
        <View
            width={'100%'}
            padding={'size-300'}
            height={'size-3400'}
            backgroundColor={'gray-50'}
            UNSAFE_style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
            }}
        >
            <Loading size='M' mode='inline' />
        </View>
    );
};
