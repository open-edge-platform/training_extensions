// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Text, View } from '@geti/ui';
import { CheckCircleOutlined } from '@geti/ui/icons';

import { $api } from '../../../../../api/client';
import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { formatBytes } from '../../../../../shared/util';
import { isInvalidStagedFile } from '../../util';

import classes from './import-job-done.module.scss';

type ImportJobDoneProps = {
    size: number;
    fileName: string;
    stagedDatasetId: string;
};

export const ImportJobDone = ({ fileName, size, stagedDatasetId }: ImportJobDoneProps) => {
    const { deleteImportEntry } = useImportDatasetToProject();
    const deleteFileMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}');

    const handleClose = () => {
        deleteFileMutation.mutate(
            { params: { path: { staged_dataset_id: stagedDatasetId } } },
            {
                onSuccess: () => {
                    deleteImportEntry(stagedDatasetId);
                },
                onError: (error) => {
                    isInvalidStagedFile(error) && deleteImportEntry(stagedDatasetId);
                },
            }
        );
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
