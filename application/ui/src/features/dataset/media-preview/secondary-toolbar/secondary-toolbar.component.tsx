// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Button, ButtonGroup, Flex, Text } from '@geti/ui';
import { Checkmark, CloseSemiBold } from '@geti/ui/icons';

import type { Media } from '../../../../constants/shared-types';
import { useProject } from '../../../../hooks/api/project.hook';
import type { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { Labels } from '../../../annotator/labels/labels.component';
import { isClassificationTask } from '../../../project/task-type-guards';
import { DeleteMediaItem } from '../../gallery/delete-media-item/delete-media-item.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';
import { useSubmitPredictions } from '../use-submit-predictions.hook';
import { AnnotatorModes } from './annotator-modes/annotator-modes-toggle.component';
import { getNextItem } from './util';

import classes from './secondary-toolbar.module.scss';

type SecondaryToolbarProps = {
    items: Media[];
    mediaItem: Media;
    mode: AnnotatorMode;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    onModeChange: (mode: AnnotatorMode) => void;
    onAcceptPrediction: () => void;
};

export const SecondaryToolbar = ({
    items,
    mediaItem,
    mode,
    onClose,
    onSelectedMediaItem,
    onModeChange,
    onAcceptPrediction,
}: SecondaryToolbarProps) => {
    const { data: selectedProject } = useProject();

    const { canSubmit, isSaving, submit } = useSubmitPredictions({ onSuccess: onAcceptPrediction });

    const isMultiLabel = selectedProject.task.exclusive_labels === false;
    const isClassification = isClassificationTask(selectedProject.task.task_type);

    const handleDeleteItem = ([deletedItem]: string[], totalItems: number) => {
        const deletedIndex = items.findIndex((item) => item.id === deletedItem);
        const nextItem = getNextItem(totalItems - 1, deletedIndex);

        onSelectedMediaItem(items[nextItem]);
    };

    return (
        <Flex
            width={'100%'}
            height={'100%'}
            alignItems={'center'}
            justifyContent={'space-between'}
            UNSAFE_className={classes.secondaryToolbarContainer}
        >
            <Toolbar.Container>
                <Toolbar.Section>
                    <AnnotatorModes mode={mode} onModeChange={onModeChange} />
                </Toolbar.Section>
            </Toolbar.Container>
            <Toolbar.Container>
                <Toolbar.Section>
                    <Labels isClassification={isClassification} isMultiLabel={isMultiLabel} />
                </Toolbar.Section>
            </Toolbar.Container>
            <Toolbar.Container>
                <Toolbar.Section>
                    <ButtonGroup>
                        <DeleteMediaItem
                            itemsIds={[String(mediaItem.id)]}
                            onDeleted={([deletedItem]: string[]) => handleDeleteItem([deletedItem], items.length - 1)}
                        />
                        <Button
                            variant='accent'
                            onPress={submit}
                            isPending={isSaving}
                            marginStart={'size-200'}
                            isDisabled={!canSubmit || isSaving}
                        >
                            {mode === 'annotation' ? (
                                'Submit'
                            ) : (
                                <>
                                    <Checkmark />
                                    <Text>Confirm prediction</Text>
                                </>
                            )}
                        </Button>

                        <ActionButton
                            isQuiet
                            onPress={onClose}
                            isDisabled={isSaving}
                            marginStart={'size-100'}
                            UNSAFE_className={classes.closeButton}
                        >
                            <CloseSemiBold width={14} height={14} />
                            <Text>Close</Text>
                        </ActionButton>
                    </ButtonGroup>
                </Toolbar.Section>
            </Toolbar.Container>
        </Flex>
    );
};
