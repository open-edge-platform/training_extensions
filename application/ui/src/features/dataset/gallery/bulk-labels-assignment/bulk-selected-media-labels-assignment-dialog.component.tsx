// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    dimensionValue,
    Divider,
    Flex,
    Heading,
    Text,
} from '@geti-ui/ui';
import { useProject } from 'hooks/api/project.hook';

import { useProjectLabelsWithEmptyLabel } from '../../../../shared/annotator/labels';
import { isMultiLabelClassificationTask } from '../../../project/task-type-guards';
import { useBulkAssignLabel } from './api/use-bulk-assign-label';
import { LabelsList } from './labels-list/labels-list.component';

type BulkSelectedMediaLabelsAssignmentProps = {
    onClose: () => void;
    onAssign: (labelIds: string[]) => Promise<void>;
    isAssignPending: boolean;
};

const BulkSelectedMediaLabelsAssignment = ({
    onClose,
    isAssignPending,
    onAssign,
}: BulkSelectedMediaLabelsAssignmentProps) => {
    const projectLabels = useProjectLabelsWithEmptyLabel();
    const { data: project } = useProject();
    const isMultiLabelClassification = isMultiLabelClassificationTask(project.task);

    const [selectedLabels, setSelectedLabels] = useState<Set<string>>(() => new Set([]));

    const isAssignDisabled = selectedLabels.size === 0 || isAssignPending;

    const handleAssign = async () => {
        await onAssign(Array.from(selectedLabels));
    };

    return (
        <Dialog height={'65vh'}>
            <Heading>Label assignment</Heading>
            <Divider />
            <Content>
                <Flex direction={'column'} gap={'size-100'} height={'100%'} minHeight={0}>
                    <Text>Choose the label(s) to assign to the selected images, then click {"'Assign'"}.</Text>
                    <Divider size={'S'} marginY={'size-100'} />
                    <LabelsList
                        ariaLabel={'Labels to assign'}
                        labels={projectLabels}
                        selectedLabels={selectedLabels}
                        onSelectedLabelsChange={setSelectedLabels}
                        isMultiple={isMultiLabelClassification}
                    />
                    <Text UNSAFE_style={{ lineHeight: dimensionValue('size-225') }}>
                        The selected labels apply only to images, videos (if any) remain intact.
                    </Text>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onClose}>
                    Cancel
                </Button>
                <Button
                    variant={'accent'}
                    onPress={handleAssign}
                    isDisabled={isAssignDisabled}
                    isPending={isAssignPending}
                >
                    Assign
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

type BulkSelectedMediaAssignmentDialogProps = {
    isVisible: boolean;
    onClose: () => void;
    selectedImagesIds: string[];
};

export const BulkSelectedMediaLabelsAssignmentDialog = ({
    isVisible,
    onClose,
    selectedImagesIds,
}: BulkSelectedMediaAssignmentDialogProps) => {
    const bulkAssignLabel = useBulkAssignLabel();

    const handleAssign = async (labelIds: string[]) => {
        await bulkAssignLabel.mutate(selectedImagesIds, labelIds);
        onClose();
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isVisible && (
                <BulkSelectedMediaLabelsAssignment
                    onClose={onClose}
                    isAssignPending={bulkAssignLabel.isPending}
                    onAssign={handleAssign}
                />
            )}
        </DialogContainer>
    );
};
