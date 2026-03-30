// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Content, Flex, Heading, IllustratedMessage, Text, View, type DropZoneProps } from '@geti-ui/ui';
import { DropZone } from 'react-aria-components';

import { ReactComponent as DropFiles } from '../../../assets/drop-files.svg';
import { getFilesFromDropEvent } from '../../../shared/drop-zone.utils';

import classes from './drop-zone.component.module.scss';

type DatasetDropZoneProps = {
    children: ReactNode;
    onFilesDropped?: (files: File[]) => void | Promise<void>;
};

type DropEvent = Parameters<NonNullable<DropZoneProps['onDrop']>>[0];

export const DatasetDropZone = ({ children, onFilesDropped }: DatasetDropZoneProps) => {
    const handleDrop = async (event: DropEvent) => {
        if (onFilesDropped === undefined) {
            return;
        }

        const files = await getFilesFromDropEvent(event);

        if (files.length > 0) {
            void onFilesDropped(files);
        }
    };

    return (
        <DropZone onDrop={handleDrop} className={classes.dropZone}>
            {(dropZoneState) => (
                <View UNSAFE_className={classes.container}>
                    {children}

                    {dropZoneState.isDropTarget && (
                        <Flex alignItems={'center'} justifyContent={'center'} UNSAFE_className={classes.dropOverlay}>
                            <IllustratedMessage UNSAFE_className={classes.dropSurface}>
                                <DropFiles />

                                <Content>
                                    <Flex alignItems={'center'} direction={'column'} gap={'size-100'}>
                                        <Heading level={2}>Drop media files here</Heading>
                                        <Text UNSAFE_className={classes.dropMessage}>
                                            Images and videos will be uploaded to this dataset.
                                        </Text>
                                    </Flex>
                                </Content>
                            </IllustratedMessage>
                        </Flex>
                    )}
                </View>
            )}
        </DropZone>
    );
};
