// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import {
    AriaDropZone,
    Content,
    Flex,
    Heading,
    IllustratedMessage,
    Text,
    toast,
    View,
    type SpectrumDropZoneProps,
} from '@geti/ui';

import { ReactComponent as DropFiles } from '../../../assets/drop-files.svg';
import { getFilesFromDropEvent } from '../../../shared/drop-zone.utils';
import { isSupportedMediaFile } from './utils';

import classes from './drop-zone.component.module.scss';

type DatasetDropZoneProps = {
    children: ReactNode;
    onFilesDropped?: (files: File[]) => void | Promise<void>;
};

type DropEvent = Parameters<NonNullable<SpectrumDropZoneProps['onDrop']>>[0];

export const DatasetDropZone = ({ children, onFilesDropped }: DatasetDropZoneProps) => {
    const handleDrop = async (event: DropEvent) => {
        if (onFilesDropped === undefined) {
            return;
        }

        const files = await getFilesFromDropEvent(event);
        const supported = files.filter(isSupportedMediaFile);

        if (supported.length < files.length) {
            toast({ type: 'neutral', message: 'Some files were skipped because their format is not supported.' });
        }

        if (supported.length > 0) {
            void onFilesDropped(supported);
        }
    };

    return (
        <AriaDropZone onDrop={handleDrop} className={classes.dropZone}>
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
        </AriaDropZone>
    );
};
