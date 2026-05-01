;
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useGetDatasetItems } from 'hooks/use-get-dataset-items.hook';





const MIN_NUMBER_OF_ANNOTATED_ITEMS = 3;
const listFormatter = new Intl.ListFormat('en', { style: 'long', type: 'conjunction' });
const pluralRules = new Intl.PluralRules('en');

const pluralizeItems = (count: number) => (pluralRules.select(count) === 'one' ? 'item' : 'items');
const conjugateToBe = (count: number) => (pluralRules.select(count) === 'one' ? 'is' : 'are');

export const useTrainModelDisabledReason = () => {
    const { totalCount, isPending: isTotalPending } = useGetDatasetItems({ annotationStatus: 'with_annotations' });
    const { totalCount: trainingSubsetSize, isPending: isTrainingPending } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subset: 'training',
    });
    const { totalCount: testingSubsetSize, isPending: isTestingPending } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subset: 'testing',
    });
    const { totalCount: validationSubsetSize, isPending: isValidationPending } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subset: 'validation',
    });
    const { totalCount: reviewedUnassignedSubsetSize, isPending: isReviewedUnassignedPending } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subset: 'unassigned',
    });
    const { totalCount: unassignedSubsetSize, isPending: isUnassignedPending } = useGetDatasetItems({
        subset: 'unassigned',
    });

    if (
        isTotalPending ||
        isTrainingPending ||
        isTestingPending ||
        isValidationPending ||
        isReviewedUnassignedPending ||
        isUnassignedPending
    ) {
        return { reason: undefined };
    }

    if (totalCount < MIN_NUMBER_OF_ANNOTATED_ITEMS) {
        return {
            reason:
                'In order to train a model, you need to annotate at least 3 items in your dataset, although we ' +
                'recommend annotating several more for better results.',
        };
    }

    const subsetSizes = [
        { name: 'training', value: trainingSubsetSize },
        { name: 'validation', value: validationSubsetSize },
        { name: 'testing', value: testingSubsetSize },
    ];

    const emptySubsets = subsetSizes.filter(({ value }) => value === 0);

    if (emptySubsets.length === 0 || emptySubsets.length <= reviewedUnassignedSubsetSize) {
        return { reason: undefined };
    }

    const emptySubsetNames = emptySubsets.map(({ name }) => name);
    const emptySubsetText =
        emptySubsetNames.length === 1
            ? `${emptySubsetNames[0]} subset is`
            : `${listFormatter.format(emptySubsetNames)} subsets are`;

    const unannotatedUnassignedSize = unassignedSubsetSize - reviewedUnassignedSubsetSize;

    let assignmentDetail: string;

    if (reviewedUnassignedSubsetSize > 0 && unannotatedUnassignedSize > 0) {
        assignmentDetail =
            `there are ${reviewedUnassignedSubsetSize} reviewed ${pluralizeItems(reviewedUnassignedSubsetSize)} ready` +
            ' to assign and ' +
            `${unannotatedUnassignedSize} ${pluralizeItems(unannotatedUnassignedSize)} that still need annotation ` +
            'before they can be assigned';
    } else if (reviewedUnassignedSubsetSize > 0) {
        assignmentDetail =
            `there are ${reviewedUnassignedSubsetSize} reviewed ` +
            `${pluralizeItems(reviewedUnassignedSubsetSize)} left to assign`;
    } else if (unannotatedUnassignedSize > 0) {
        assignmentDetail =
            `there ${conjugateToBe(unannotatedUnassignedSize)} ${unannotatedUnassignedSize} ` +
            `${pluralizeItems(unannotatedUnassignedSize)} that still need annotation before they ` +
            'can be assigned';
    } else {
        assignmentDetail = 'there are no unassigned items available to redistribute';
    }

    return {
        reason:
            'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
            `This condition is currently not satisfiable, because the ${emptySubsetText} empty and ` +
            `${assignmentDetail}.`,
    };
};
