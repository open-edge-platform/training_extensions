// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Flex, Heading, Text } from '@geti/ui';
import { Info } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { MediaDTO } from '../../../../constants/shared-types';
import { useProjectLabelsWithEmptyLabel } from '../../../../shared/annotator/labels';
import { LabelsList } from './labels-list/labels-list.component';

type BulkLabelsAssignmentDialogContentProps = {
    onClose: () => void;
    onSkip: () => void;
    onAccept: (files: File[]) => Promise<MediaDTO[]>;
    isMultiLabelClassification: boolean;
};

const BulkLabelsAssignmentDialogContent = ({
    onClose,
    onSkip,
    onAccept,
    isMultiLabelClassification,
}: BulkLabelsAssignmentDialogContentProps) => {
    const projectLabels = useProjectLabelsWithEmptyLabel();

    const [selectedLabels, setSelectedLabels] = useState<Set<string>>(() => new Set([]));

    const handleAccept = () => {
        onAccept([]);
    };

    return (
        <Dialog height={'65vh'}>
            <Heading>Label assignment</Heading>
            <Divider />
            <Content>
                <Flex direction={'column'} gap={'size-100'} height={'100%'} minHeight={0}>
                    <LabelsList
                        ariaLabel={'Labels to assign'}
                        labels={projectLabels}
                        selectedLabels={selectedLabels}
                        onSelectedLabelsChange={setSelectedLabels}
                        isMultiple={isMultiLabelClassification}
                    />
                    <Flex alignItems={'center'} gap={'size-50'}>
                        <Info />
                        <Text>Labeling applies only to images</Text>
                    </Flex>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onClose}>
                    Cancel upload
                </Button>
                <Button variant={'secondary'} onPress={onSkip}>
                    Skip
                </Button>
                <Button variant={'accent'} onPress={handleAccept}>
                    Accept
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

type BulkLabelsAssignmentDialogProps = {
    files: File[];
    onClose: () => void;
    isMultiLabelClassification: boolean;
    onDatasetItemsUpload: (files: File[]) => Promise<MediaDTO[]>;
};

export const BulkLabelsAssignmentDialog = ({
    files,
    onClose,
    onDatasetItemsUpload,
    isMultiLabelClassification,
}: BulkLabelsAssignmentDialogProps) => {
    const isVisible = !isEmpty(files);

    const handleSkip = async () => {
        await onDatasetItemsUpload(files);
        onClose();
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isVisible && (
                <BulkLabelsAssignmentDialogContent
                    onClose={onClose}
                    onSkip={handleSkip}
                    onAccept={onDatasetItemsUpload}
                    isMultiLabelClassification={isMultiLabelClassification}
                />
            )}
        </DialogContainer>
    );
};
