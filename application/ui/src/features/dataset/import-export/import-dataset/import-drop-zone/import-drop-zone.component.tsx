// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, DropZone, FileTrigger, Flex, IllustratedMessage, Link as SpectrumLink, Text } from '@geti/ui';
import { LinkOut } from '@geti/ui/icons';

import { ReactComponent as EmptyDataset } from '../../../../../assets/drop-files.svg';
import { ImportDatasetState } from '../util';

import classes from './import-drop-zone.module.scss';

type ImportDropZoneProps = {
    onNextStep: (step: ImportDatasetState) => void;
};

export const ImportDropZone = ({ onNextStep }: ImportDropZoneProps) => {
    return (
        <DropZone onDrop={() => onNextStep('process')}>
            <IllustratedMessage>
                <EmptyDataset />

                <Content>
                    <Flex alignItems={'center'} direction={'column'} gap={'size-100'}>
                        <Text>Drop the dataset .zip file here</Text>

                        <FileTrigger onSelect={() => onNextStep('process')}>
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
