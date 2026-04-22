// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties } from 'react';

import { ActionButton, Content, Dialog, DialogTrigger, dimensionValue } from '@geti/ui';
import { Filter } from '@geti/ui/icons';
import { useLabelsSearchParams } from 'hooks/use-labels-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';

import { MultiSelectList } from '../../../../../components/multi-select-list/multi-select-list.component';

interface MediaFilterProps {
    isDisabled?: boolean;
}

const paddingStyle = {
    '--spectrum-dialog-padding-x': dimensionValue('size-300'),
    '--spectrum-dialog-padding-y': dimensionValue('size-300'),
} as CSSProperties;

export const MediaFilter = ({ isDisabled = false }: MediaFilterProps) => {
    const labels = useProjectLabels();
    const { selectedLabelIds, setSelectedLabelIds } = useLabelsSearchParams();

    const handleSelectionChange = (selectedKeys: Set<string> | 'all') => {
        const ids = selectedKeys === 'all' ? labels.map(({ id }) => id) : Array.from(selectedKeys);

        setSelectedLabelIds(ids);
    };

    return (
        <DialogTrigger hideArrow type='popover'>
            <ActionButton isQuiet isDisabled={isDisabled} aria-label='Filter media items'>
                <Filter />
            </ActionButton>

            <Dialog width={'size-5000'} UNSAFE_style={paddingStyle} aria-label='Filter media items'>
                <Content>
                    <MultiSelectList
                        name='labels'
                        items={labels}
                        maxHeight='size-2000'
                        onSelectionChange={handleSelectionChange}
                        defaultSelectedKeys={new Set(selectedLabelIds)}
                    />
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
