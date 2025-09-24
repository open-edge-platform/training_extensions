// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';

import { ProjectDetails } from '../../features/project/details/project-details.component';
import Background from './../../assets/background.png';

export const EditProject = () => {
    return (
        <View
            backgroundColor={'gray-100'}
            UNSAFE_style={{
                backgroundImage: `url(${Background})`,
                backgroundBlendMode: 'luminosity',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                backgroundSize: 'cover',
            }}
            gridArea={'content'}
            height='100%'
            width='100%'
        >
            <View paddingY='size-800'>
                <View
                    UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}
                    maxWidth={'1320px'}
                    marginX='auto'
                >
                    <ProjectDetails />
                </View>
            </View>
        </View>
    );
};
