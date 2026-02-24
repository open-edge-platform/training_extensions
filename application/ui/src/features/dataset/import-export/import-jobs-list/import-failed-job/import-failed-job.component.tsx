// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';

import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { usePrepareImportDataset } from '../../../../../hooks/localStorage/use-prepare-import-dataset.hook';
import { formatBytes } from '../../../../../shared/util';

type ImportFailedJobProps = {
    size: number;
    fileName: string;
    job: PrepareImportDatasetJob;
};

export const ImportFailedJob = ({ job, fileName, size }: ImportFailedJobProps) => {
    const { removeLsPreparingImport } = usePrepareImportDataset();

    const handleClose = () => {
        removeLsPreparingImport();
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>
                    Import dataset - {fileName} - {formatBytes(size)}
                </Text>

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close import dataset status'
                        onPress={handleClose}
                    >
                        Close
                    </Button>
                </Flex>
            </Flex>

            <Text>{job.message}</Text>
            <Divider size='S' marginY='size-150' />
            <Text>{job.error}</Text>
        </View>
    );
};
