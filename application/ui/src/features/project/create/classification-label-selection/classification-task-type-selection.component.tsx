// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Radio, RadioGroup, Text } from '@geti/ui';

import styles from './classification-task-type-selection.module.scss';

export type ClassificationTaskType = 'single-label' | 'multi-label';

type ClassificationTaskTypeItemProps = {
    type: ClassificationTaskType;
    label: string;
    description: string;
    onSelect: (type: ClassificationTaskType) => void;
};

const ClassificationTaskTypeItem = ({ type, label, description, onSelect }: ClassificationTaskTypeItemProps) => {
    return (
        <div onClick={() => onSelect(type)} className={styles.itemType}>
            <Radio value={type}>{label}</Radio>

            <Text UNSAFE_className={styles.typeDescription}>{description}</Text>
        </div>
    );
};

type ClassificationTaskSelectionProps = {
    selectedType: ClassificationTaskType;
    onSelectedTypeChange: (type: ClassificationTaskType) => void;
};

export const ClassificationTaskSelection = ({
    selectedType,
    onSelectedTypeChange,
}: ClassificationTaskSelectionProps) => {
    return (
        <Flex direction={'column'} gap={'size-250'}>
            <Text UNSAFE_className={styles.title}>What classification type should the model use?</Text>
            <RadioGroup
                isEmphasized
                aria-label={'Classification type'}
                value={selectedType}
                onChange={(value) => onSelectedTypeChange(value as ClassificationTaskType)}
            >
                <Flex justifyContent={'center'}>
                    <Flex gap={'size-200'} width={'80%'}>
                        <ClassificationTaskTypeItem
                            type={'single-label'}
                            label={'Single-label'}
                            description={'Assign one label from mutually exclusive labels'}
                            onSelect={onSelectedTypeChange}
                        />
                        <ClassificationTaskTypeItem
                            type={'multi-label'}
                            label={'Multi-label'}
                            description={'Assign one or more labels from non-mutually exclusive labels'}
                            onSelect={onSelectedTypeChange}
                        />
                    </Flex>
                </Flex>
            </RadioGroup>
        </Flex>
    );
};
