// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Heading, View } from '@geti/ui';

import IntelBrandedLoadingGif from '../../assets/intel-loading.webp';

export const AnnotatorLoading = ({ isLoading }: { isLoading: boolean }) => {
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
                {/*  eslint-disable-next-line jsx-a11y/img-redundant-alt */}
                <img
                    src={IntelBrandedLoadingGif}
                    // eslint-disable-next-line jsx-a11y/no-noninteractive-element-to-interactive-role
                    role='progressbar'
                    alt='Processing image'
                    style={{
                        width: 300,
                        height: 300,
                    }}
                />
                <Heading
                    level={1}
                    UNSAFE_style={{
                        textShadow: '1px 1px 2px black, 1px 1px 2px white',
                    }}
                >
                    {isLoading && 'Processing image, please wait...'}
                </Heading>
            </Flex>
        </View>
    );
};
