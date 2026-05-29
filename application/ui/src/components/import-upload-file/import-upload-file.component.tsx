// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, DropZone, FileTrigger, Flex, Heading, IllustratedMessage, Text, toast } from '@geti/ui';
import { LinkOut } from '@geti/ui/icons';

import { $api } from '../../api/client';
import { ReactComponent as EmptyDataset } from '../../assets/drop-files.svg';
import { Link } from '../../platform/components/link.component';
import { getFilesFromDropEvent } from '../../shared/drop-zone.utils';
import { formatToFileArray, isSupportedDatasetZip } from './util';

import classes from './import-upload-file.module.scss';

export type FileUploadedResponse = { size: number; fileName: string; prepareJobId: string; stagedDatasetId: string };

type ImportUploadFileProps = {
    formatOptions: string;
    onFileUploaded: (data: FileUploadedResponse) => void;
};

export const ImportUploadFile = ({ formatOptions, onFileUploaded }: ImportUploadFileProps) => {
    const stagedDatasetMutation = $api.useMutation('post', '/api/staged_datasets');
    const prepareImportJobMutation = $api.useMutation('post', '/api/jobs');

    const handleLoadingFile = (files: File[]) => {
        const hasMultipleFiles = files.length > 1;

        if (hasMultipleFiles) {
            toast({
                message: 'Adding folders or multiple files is not allowed. Please load a single file.',
                type: 'error',
            });
            return;
        }

        if (!isSupportedDatasetZip(files[0])) {
            toast({
                message: 'Unsupported file format. Please upload a valid .zip file.',
                type: 'error',
            });
            return;
        }

        handleImportPrepare(files[0]);
    };

    const handleImportPrepare = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);

        // @ts-expect-error There is an incorrect type in OpenAPI
        const stagedDataset = await stagedDatasetMutation.mutateAsync({ body: formData });

        const prepareImportJob = await prepareImportJobMutation.mutateAsync({
            body: {
                job_type: 'prepare_dataset_for_import',
                staged_dataset_id: stagedDataset.id,
            },
        });

        onFileUploaded({
            size: file.size,
            fileName: file.name,
            prepareJobId: prepareImportJob.job_id,
            stagedDatasetId: stagedDataset.id,
        });
    };

    const isPending = stagedDatasetMutation.isPending || prepareImportJobMutation.isPending;

    return (
        <DropZone
            isFilled={stagedDatasetMutation.isSuccess}
            onDrop={async (event) => handleLoadingFile(await getFilesFromDropEvent(event))}
        >
            <IllustratedMessage maxHeight={'size-4600'}>
                <EmptyDataset />

                <Content>
                    {isPending && (
                        <Flex alignItems={'center'} direction={'column'} gap={'size-100'}>
                            <Heading level={1}>Uploading...</Heading>
                            <Text>Dataset is being uploaded</Text>
                        </Flex>
                    )}

                    {!isPending && (
                        <Flex alignItems={'center'} direction={'column'} gap={'size-100'}>
                            <Text>Drop the dataset .zip file here</Text>

                            <FileTrigger
                                data-testid='upload-zip-file'
                                onSelect={(data) => handleLoadingFile(formatToFileArray(data))}
                            >
                                <Button marginY={'size-200'} maxWidth={'size-1000'} variant={'accent'}>
                                    Upload
                                </Button>
                            </FileTrigger>

                            <Text UNSAFE_className={classes.formatOptions}>({formatOptions}).zip</Text>

                            <Link href='/' target='_blank' UNSAFE_className={classes.link}>
                                Learn more about the different formats
                                <LinkOut size='XS' />
                            </Link>
                        </Flex>
                    )}
                </Content>
            </IllustratedMessage>
        </DropZone>
    );
};
