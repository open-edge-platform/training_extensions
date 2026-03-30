// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Flex, Heading, Text, toast } from '@geti/ui';
import { Info } from '@geti/ui/icons';
import { dimensionValue } from '@react-spectrum/utils';
import { useProject } from 'hooks/api/project.hook';
import { isEmpty } from 'lodash-es';

import { useProjectLabelsWithEmptyLabel } from '../../../../shared/annotator/labels';
import { isImage } from '../../../../shared/media-item-utils';
import { isMultiLabelClassificationTask } from '../../../project/task-type-guards';
import { useMediaUpload } from '../../api/use-media-upload';
import { useAssignLabel } from './api/use-assign-label';
import { LabelsList } from './labels-list/labels-list.component';
import { useBulkUploadAndAssignLabel } from './use-bulk-upload-and-assign-label';

type BulkLabelsAssignmentDialogContentProps = {
    onClose: () => void;
    onSkip: () => void;
    isSkipPending: boolean;
    onContinue: (labelIds: string[]) => Promise<void>;
    isContinuePending: boolean;
    isMultiLabelClassification: boolean;
};

const BulkLabelsAssignmentDialogContent = ({
    onClose,
    onSkip,
    onContinue,
    isSkipPending,
    isContinuePending,
    isMultiLabelClassification,
}: BulkLabelsAssignmentDialogContentProps) => {
    const projectLabels = useProjectLabelsWithEmptyLabel();

    const [selectedLabels, setSelectedLabels] = useState<Set<string>>(() => new Set([]));

    const isContinueDisabled = selectedLabels.size === 0 || isContinuePending;

    const handleContinue = async () => {
        await onContinue(Array.from(selectedLabels));
    };

    return (
        <Dialog height={'65vh'}>
            <Heading>Label assignment</Heading>
            <Divider />
            <Content>
                <Flex direction={'column'} gap={'size-100'} height={'100%'} minHeight={0}>
                    <Text>
                        Choose the label(s) to assign to the uploaded images, then click {"'Continue'"}. If you instead
                        prefer to annotate the images at a later time, choose {"'Skip'"}.
                    </Text>
                    <Divider size={'S'} marginY={'size-100'} />
                    <LabelsList
                        ariaLabel={'Labels to assign'}
                        labels={projectLabels}
                        selectedLabels={selectedLabels}
                        onSelectedLabelsChange={setSelectedLabels}
                        isMultiple={isMultiLabelClassification}
                    />
                    <Flex gap={'size-50'}>
                        <Info />
                        <Text UNSAFE_style={{ lineHeight: dimensionValue('size-225') }}>
                            The selected labels apply only to images, videos (if any) will be uploaded without
                            annotations.
                        </Text>
                    </Flex>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onClose}>
                    Cancel upload
                </Button>
                <Button variant={'secondary'} onPress={onSkip} isPending={isSkipPending} isDisabled={isSkipPending}>
                    Skip
                </Button>
                <Button
                    variant={'accent'}
                    onPress={handleContinue}
                    isDisabled={isContinueDisabled}
                    isPending={isContinuePending}
                >
                    Continue
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

type BulkLabelsAssignmentDialogProps = {
    files: File[];
    onClose: () => void;
};

export const BulkLabelsAssignmentDialog = ({ files, onClose }: BulkLabelsAssignmentDialogProps) => {
    const isVisible = !isEmpty(files);
    const { data: project } = useProject();
    const isMultiLabelClassification = isMultiLabelClassificationTask(project.task);
    const { uploadMedia, uploadProgress } = useMediaUpload();

    const assignLabel = useAssignLabel();

    const handleSkip = async () => {
        await uploadMedia(files);
        onClose();
    };

    const handleAccept = async (labelIds: string[]) => {
        const mediaItems = await uploadMedia(files);
        const mediaItemImages = mediaItems.filter(isImage);

        const result = await Promise.allSettled(mediaItemImages.map((media) => assignLabel.mutate(media.id, labelIds)));

        const successfulMediaItems = result.filter(({ status }) => status === 'fulfilled');
        const failedMediaItems = result.filter(({ status }) => status === 'rejected');

        if (failedMediaItems.length === 0) {
            toast({
                type: 'success',
                message: `Successfully assigned labels to all ${successfulMediaItems.length} media items`,
            });
        } else if (successfulMediaItems.length === 0) {
            toast({
                type: 'error',
                message: `Failed to assign labels to all ${failedMediaItems.length} media items`,
            });
        } else {
            toast({
                type: 'info',
                message:
                    `Assigned labels to ${successfulMediaItems.length} of ${mediaItems.length} media items ` +
                    `(${failedMediaItems.length} failed)`,
            });
        }

        assignLabel.invalidateQueries();

        onClose();
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isVisible && (
                <BulkLabelsAssignmentDialogContent
                    onClose={onClose}
                    onSkip={handleSkip}
                    onContinue={handleAccept}
                    isContinuePending={uploadProgress.isUploading || assignLabel.isPending}
                    isSkipPending={uploadProgress.isUploading}
                    isMultiLabelClassification={isMultiLabelClassification}
                />
            )}
        </DialogContainer>
    );
};
