// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';

import { ProjectDetails } from '../../features/project/details/project-details.component';

import styles from '../../features/project/project-background.module.scss';

export const ViewProject = () => {
    return (
        <View UNSAFE_className={styles.projectBackground} gridArea={'content'} height='100%' width='100%'>
            <View paddingY='size-800'>
                <View maxWidth={'1320px'} marginX='auto'>
                    <ProjectDetails />
                </View>
            </View>
        </View>
    );
};
