// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Text } from '@geti/ui';
import { Alert } from '@geti/ui/icons';
import { isNil } from 'lodash-es';

import { AnnotationType } from '../../../../../../constants/shared-types';
import { useProject } from '../../../../../../hooks/api/project.hook';

import classes from './format-warning.module.scss';

type FormatWarningProps = {
    annotationType?: AnnotationType;
};

const getMessage = (taskType: string, annotationType?: AnnotationType) => {
    if (annotationType === 'bounding_box' && taskType === 'instance_segmentation') {
        // eslint-disable-next-line max-len
        return 'Imported dataset uses bounding box annotations, but your project’s dataset uses polygons. Annotations will be automatically converted to polygons to keep everything compatible.';
    }

    if (annotationType === 'polygon' && taskType === 'detection') {
        // eslint-disable-next-line max-len
        return 'Imported dataset uses polygon annotations, but your project’s dataset uses bounding boxes. Annotations will be automatically converted to bounding boxes to keep everything compatible.';
    }

    return null;
};

export const FormatWarning = ({ annotationType }: FormatWarningProps) => {
    const { data: selectedProject } = useProject();

    const message = getMessage(selectedProject?.task?.task_type, annotationType);

    if (isNil(annotationType) || isNil(message)) {
        return null;
    }

    return (
        <>
            <Divider size={'S'} marginY={'size-125'} />

            <Flex gap={'size-125'}>
                <div className={classes.iconContainer}>
                    <Alert width={24} height={24} />
                </div>

                <Text>{message}</Text>
            </Flex>
        </>
    );
};
