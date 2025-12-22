// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useAnnotator } from 'src/shared/annotator/annotator-provider.component';

import { Label } from '../../../constants/shared-types';
import { AddLabel } from './add-label/add-label.component';
import { LabelListItem } from './label-list-item/label-list-item.component';

interface LabelsListProps {
    labels: Label[];
    selectedLabelId: string | null;
    setSelectedLabelId: (label: string) => void;
}

const LabelsList = ({ labels, selectedLabelId, setSelectedLabelId }: LabelsListProps) => {
    return labels.map((label) => (
        <LabelListItem
            key={label.id}
            label={label}
            onSelect={() => setSelectedLabelId(label.id)}
            isSelected={selectedLabelId === label.id}
            existingLabels={labels}
        />
    ));
};

export const Labels = () => {
    const { selectedLabelId, setSelectedLabelId, labels } = useAnnotator();

    return (
        <Flex height={'100%'} alignItems={'center'} width={'100%'}>
            <Flex margin={'size-50'} wrap={'wrap'} width={'100%'} alignItems={'center'} gap={'size-100'}>
                <LabelsList labels={labels} selectedLabelId={selectedLabelId} setSelectedLabelId={setSelectedLabelId} />
                <Flex alignSelf={'flex-end'} flex={1} justifyContent={'end'} alignItems={'center'}>
                    <AddLabel existingLabels={labels} />
                </Flex>
            </Flex>
        </Flex>
    );
};
