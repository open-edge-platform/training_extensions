// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useProjectLabelsWithEmptyLabel } from '../../../../../../shared/annotator/labels';
import { LabelsList } from './labels-list.component';

type BulkLabelAssignmentDialogContentProps = {
    onClose: () => void;
    onSkip: () => void;
    onAccept: (labelIds: string[]) => void;
    isMultiLabelClassification: boolean;
};

const BulkLabelAssignmentDialogContent = ({
    onClose,
    onSkip,
    onAccept,
    isMultiLabelClassification,
}: BulkLabelAssignmentDialogContentProps) => {
    const projectLabels = useProjectLabelsWithEmptyLabel();

    const [selectedLabels, setSelectedLabels] = useState<Set<string>>(() => new Set([]));

    const handleAccept = () => {
        onAccept(Array.from(selectedLabels));
    };

    return (
        <Dialog minHeight={'size-6000'}>
            <Heading>Assign the label(s) to the uploaded dataset items</Heading>
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
                <Button variant={'primary'} onPress={handleAccept}>
                    Accept
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

type LabelAssignmentDialogProps = {
    files: File[];
    onClose: () => void;
    isMultiLabelClassification: boolean;
    onDatasetItemsUpload: (files: File[]) => Promise<void>;
};

export const LabelAssignmentDialog = ({
    files,
    onClose,
    onDatasetItemsUpload,
    isMultiLabelClassification,
}: LabelAssignmentDialogProps) => {
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
                <BulkLabelAssignmentDialogContent
                    onClose={onClose}
                    onSkip={handleSkip}
                    onAccept={handleAccept}
                    isMultiLabelClassification={isMultiLabelClassification}
                />
            )}
        </DialogContainer>
    );
};
