// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { CheckCircleOutlined } from '@geti/ui/icons';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { formatBytes } from '../../../shared/util';

import classes from './import-job-done.module.scss';

type ImportJobDoneProps = {
    size: number;
    fileName: string;
    stagedDatasetId: string;
    deleteEntry: () => void;
};

export const ImportJobDone = ({ fileName, size, stagedDatasetId, deleteEntry }: ImportJobDoneProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const handleClose = () => {
        deleteFileMutation.mutate();
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

            <Text>{fileName} file has been imported successfully</Text>
            <Divider size='S' marginY='size-150' />

            <Flex alignItems='center' gap='size-100'>
                <CheckCircleOutlined className={classes.checkIcon} width={16} height={16} />

                <Text>Ready</Text>
            </Flex>
        </View>
    );
};
