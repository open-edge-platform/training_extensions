// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Button, ButtonGroup, dimensionValue, Flex, Key, Text } from '@geti/ui';
import { Checkmark, CloseSemiBold } from '@geti/ui/icons';
import { useQueryClient, type QueryClient } from '@tanstack/react-query';
import { isEmpty } from 'lodash-es';

import type { Label, Media } from '../../../../constants/shared-types';
import { useProject } from '../../../../hooks/api/project.hook';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../../shared/annotator/select-annotation-provider.component';
import { isClassificationTask } from '../../../project/task-type-guards';
import { DeleteMediaItem } from '../../gallery/delete-media-item/delete-media-item.component';
import { useSelectedData } from '../../selected-data-provider.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';
import { AnnotatorModes } from './annotator-modes/annotator-modes-toggle.component';
import type { AnnotatorMode } from './annotator-modes/mode';
import { LabelPicker } from './label-picker.component';
import { useSecondaryToolbarState } from './use-secondary-toolbar-state.hook';
import { toggleLabel } from './util';

import styles from './secondary-toolbar.module.scss';

type SecondaryToolbarProps = {
    items: Media[];
    mediaItem: Media;
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;

    mode: AnnotatorMode;
    onModeChange: (mode: AnnotatorMode) => void;
};

const getNextItem = (totalItems: number, newIndex: number) => {
    return Math.min(totalItems, newIndex + 1);
};

const invalidateMediaItemAnnotations = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: ['get', '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations'],
    });
};

export const SecondaryToolbar = ({
    items,
    mediaItem,
    onClose,
    onSelectedMediaItem,
    mode,
    onModeChange,
}: SecondaryToolbarProps) => {
    const queryClient = useQueryClient();
    const { setMediaState } = useSelectedData();
    const { projectLabels } = useSecondaryToolbarState();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { data: selectedProject } = useProject();
    const { selectedLabel, setSelectedLabelId } = useAnnotator();

    const {
        annotations,
        isSaving,
        addAnnotations,
        updateAnnotations,
        deleteAnnotations,
        submitAnnotations,
        submitPredictions,
    } = useAnnotationActions();

    const hasAnnotations = !isEmpty(annotations);
    const isMultiLabel = selectedProject.task.exclusive_labels === false;
    const selectedIndex = items.findIndex((item) => item.id === mediaItem.id);

    const handleSubmit = async () => {
        await submitAnnotations();

        setMediaState((prev) => {
            const newState = new Map(prev);

            newState.set(String(mediaItem.id), 'accepted');

            return newState;
        });

        const nextItem = getNextItem(items.length - 1, selectedIndex);
        onSelectedMediaItem(items[nextItem]);

        const isLastItem = selectedIndex === items.length - 1;
        isLastItem && invalidateMediaItemAnnotations(queryClient);
    };

    const handleDeleteItem = ([deletedItem]: string[], totalItems: number) => {
        const deletedIndex = items.findIndex((item) => item.id === deletedItem);
        const nextItem = getNextItem(totalItems - 1, deletedIndex);

        onSelectedMediaItem(items[nextItem]);
    };

    const handleSelect = (value: Key | null) => {
        const label = projectLabels.find(({ id }) => id === value);
        const labels = label ? [label] : [];

        const updatedAnnotations = annotations
            .filter((annotation) => selectedAnnotations.has(annotation.id))
            .map((annotation) => ({ ...annotation, labels }));

        updateAnnotations(updatedAnnotations);
        setSelectedLabelId(label?.id ?? null);
    };

    const handleClassificationSelect = (value: Key | null) => {
        const label = projectLabels.find(({ id }) => id === value);
        const labels = label ? [label] : [];

        if (isEmpty(annotations)) {
            addAnnotations([{ type: 'full_image' }], labels);
        } else if (isMultiLabel) {
            updateClassificationAnnotations(labels[0]);
        } else {
            updateAnnotations(annotations.map((annotation) => ({ ...annotation, labels })));
        }

        setSelectedLabelId(label?.id ?? null);
    };

    const updateClassificationAnnotations = (newLabel: Label) => {
        const updatedAnnotations = annotations.map((annotation) => ({
            ...annotation,
            labels: toggleLabel(newLabel, annotation.labels),
        }));

        const hasNoLabels = updatedAnnotations.every(({ labels }) => isEmpty(labels));

        if (hasNoLabels) {
            deleteAnnotations(updatedAnnotations.map(({ id }) => id));
        } else {
            updateAnnotations(updatedAnnotations);
        }
    };

    return (
        <Flex
            width={'100%'}
            height={'100%'}
            alignItems={'center'}
            justifyContent={'space-between'}
            UNSAFE_style={{ paddingTop: dimensionValue('size-125') }}
        >
            <Toolbar.Container>
                <Toolbar.Section>
                    <AnnotatorModes mode={mode} onModeChange={onModeChange} />
                </Toolbar.Section>
            </Toolbar.Container>
            <Toolbar.Container>
                <Toolbar.Section>
                    <LabelPicker
                        labels={projectLabels}
                        selectedLabel={selectedLabel}
                        onSelect={
                            isClassificationTask(selectedProject.task.task_type)
                                ? handleClassificationSelect
                                : handleSelect
                        }
                    />
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
                            onPress={handleSubmit}
                            isPending={isSaving}
                            marginStart={'size-200'}
                            isDisabled={!hasAnnotations || isSaving}
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
                            UNSAFE_className={styles.closeButton}
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
