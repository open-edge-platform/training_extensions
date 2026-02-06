// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, Fragment } from 'react';

import { Divider, Flex, Text } from '@geti/ui';
import { clsx } from 'clsx';

import type { Label } from '../../../constants/shared-types';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { LabelsEditorPopover } from './labels-editor/labels-editor-popover.component';
import { useLabels } from './use-labels.hook';

import classes from './labels.module.scss';

type LabelBadgeProps = {
    label: Label;
    isSelected: boolean;
    onClick: () => void;
};

const LabelBadge = ({ label, isSelected, onClick }: LabelBadgeProps) => {
    return (
        <button
            onClick={onClick}
            style={{ '--labelBgColor': label.color } as CSSProperties}
            className={clsx(classes.badge, { [classes.selected]: isSelected })}
            aria-pressed={isSelected}
            aria-label={`Label ${label.name}`}
        >
            <Text UNSAFE_className={classes.badgeText}>{label.name}</Text>
        </button>
    );
};

type LabelsProps = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
};

export const Labels = ({ isClassification = false, isMultiLabel = false }: LabelsProps) => {
    const { labels, hasLabels, toggleLabelOnAnnotations, isLabelActive } = useLabels({
        isClassification,
        isMultiLabel,
    });

    return (
        <Flex alignItems='start' gap='size-100' minWidth={0} flex='1'>
            {hasLabels && (
                <div aria-label={'Labels'} className={classes.labelsContainer}>
                    {labels.map((label) => (
                        <Fragment key={label.id}>
                            {label.id === EMPTY_LABEL_ID && <Divider size={'S'} orientation={'vertical'} />}
                            <LabelBadge
                                label={label}
                                isSelected={isLabelActive(label)}
                                onClick={() => toggleLabelOnAnnotations(label)}
                            />
                        </Fragment>
                    ))}
                </div>
            )}
            <LabelsEditorPopover
                isClassification={isClassification}
                isMultiLabel={isMultiLabel}
                hasLabels={hasLabels}
            />
        </Flex>
    );
};
