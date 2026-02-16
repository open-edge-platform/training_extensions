// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    Content,
    DropZone,
    FileTrigger,
    Flex,
    IllustratedMessage,
    Link as SpectrumLink,
    Text,
    toast,
} from '@geti/ui';
import { LinkOut } from '@geti/ui/icons';

import { ReactComponent as EmptyDataset } from '../../../../../assets/drop-files.svg';
import { useStageDataset } from '../../../../../hooks/localStorage/use-stage-dataset.hook';
import { ImportDatasetState } from '../util';
import { formatToFileArray, getFilesFromDropEvent, isSupportedDatasetZip } from './util';

import classes from './import-drop-zone.module.scss';

type ImportDropZoneProps = {
    onNextStep: (step: ImportDatasetState) => void;
};

export const ImportDropZone = ({ onNextStep }: ImportDropZoneProps) => {
    const { addLsStagingId } = useStageDataset();

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

        addLsStagingId(`${files[0].name}-${Date.now()}`);
        onNextStep('process');
    };

    return (
        <DropZone onDrop={async (event) => handleLoadingFile(await getFilesFromDropEvent(event))}>
            <IllustratedMessage>
                <EmptyDataset />

                <Content>
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

                        <Text UNSAFE_className={classes.formatOptions}>(Geti, COCO) .zip</Text>

                        <SpectrumLink UNSAFE_className={classes.link}>
                            <a href={'/'} target={'_blank'} rel={'noopener noreferrer'}>
                                Learn more the different formats
                                <LinkOut size='XS' />
                            </a>
                        </SpectrumLink>
                    </Flex>
                </Content>
            </IllustratedMessage>
        </DropZone>
    );
};
