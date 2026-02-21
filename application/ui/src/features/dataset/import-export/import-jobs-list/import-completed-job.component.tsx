// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';

import { PrepareImportDatasetJob } from '../../../../constants/shared-types';
import { formatBytes } from './util';

type ImportCompletedJobProps = {
    size: number;
    fileName: string;
    job: PrepareImportDatasetJob;
};

export const ImportCompletedJob = ({ job: _job, size, fileName }: ImportCompletedJobProps) => {
    const handleDelete = () => {
        // Todo: implement once https://github.com/open-edge-platform/training_extensions/pull/5558 get merged
    };

    const handleContinue = () => {
        // Todo: implement once https://github.com/open-edge-platform/training_extensions/pull/5558 get merged
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>
                    Import dataset - {fileName} - {formatBytes(size)}
                </Text>

                <Divider size='S' marginY='size-150' />

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close export dataset status'
                        onPress={handleDelete}
                    >
                        Delete
                    </Button>
                    <Button aria-label='download dataset' onPress={handleContinue}>
                        Continue
                    </Button>
                </Flex>
            </Flex>

            <Divider size='S' marginY='size-150' />

            <Flex alignItems='center' gap='size-100'>
                <InfoOutline width={16} height={16} />

                <Text>Map labels for the uploaded dataset</Text>
            </Flex>
        </View>
    );
};
