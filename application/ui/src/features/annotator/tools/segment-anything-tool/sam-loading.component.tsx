// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Heading, IntelBrandedLoading, View } from '@geti/ui';

export const SAMLoading = ({ isLoading }: { isLoading: boolean }) => {
    return (
        <View
            position={'absolute'}
            left={0}
            top={0}
            right={0}
            bottom={0}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-alias-background-color-modal-overlay)',
                zIndex: 10,
            }}
        >
            <Flex direction={'column'} alignItems={'center'} justifyContent={'center'} height='100%' gap='size-100'>
                <IntelBrandedLoading />
                <Heading
                    level={1}
                    UNSAFE_style={{
                        fontSize: 'calc(var(--spectrum-global-dimension-size-200) / var(--zoom-scale, 1))',
                        textShadow: '1px 1px 2px black, 1px 1px 2px white',
                    }}
                >
                    {isLoading && 'Processing image, please wait...'}
                </Heading>
            </Flex>
        </View>
    );
};
