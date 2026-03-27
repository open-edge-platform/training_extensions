// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useProjectLabelsWithEmptyLabel } from '../../../../shared/annotator/labels';
import { LabelsList } from './labels-list/labels-list.component';

type BulkLabelsAssignmentDialogContentProps = {
    onClose: () => void;
    onSkip: () => void;
    onAccept: (labelIds: string[]) => void;
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
        onAccept(Array.from(selectedLabels));
    };

    return (
        <Dialog minHeight={'size-6000'}>
            <Heading>Label assignment</Heading>
            <Divider />
            <Content>
                <LabelsList
                    ariaLabel={'Labels to assign'}
                    labels={projectLabels}
                    selectedLabels={selectedLabels}
                    onSelectedLabelsChange={setSelectedLabels}
                    isMultiple={isMultiLabelClassification}
                />
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
    onDatasetItemsUpload: (files: File[]) => Promise<void>;
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

    const handleAccept = (_labelIds: string[]) => {
        // TODO: Send bulk label assignment request
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isVisible && (
                <BulkLabelsAssignmentDialogContent
                    onClose={onClose}
                    onSkip={handleSkip}
                    onAccept={handleAccept}
                    isMultiLabelClassification={isMultiLabelClassification}
                />
            )}
        </DialogContainer>
    );
};
