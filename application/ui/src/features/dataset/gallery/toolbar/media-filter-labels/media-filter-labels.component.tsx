// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, Dialog, DialogTrigger, Flex, PressableElement, Text } from '@geti/ui';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { isEmpty } from 'lodash-es';

import { MultiSelectList } from '../../../../../components/multi-select-list/multi-select-list.component';
import { Label } from '../../../../../constants/shared-types';
import { FilterChips } from './filter-chips.component';

import classes from './media-filter-labels.module.scss';

export const MediaFilterLabels = () => {
    const labels = useProjectLabels();
    const { selectedLabelIds, setSelectedLabelIds } = useDatasetFiltersSearchParams();

    const handleSelectionChange = (selectedKeys: Set<string> | 'all') => {
        const ids = selectedKeys === 'all' ? labels.map(({ id }) => id) : Array.from(selectedKeys);

        setSelectedLabelIds(ids);
    };

    const handleRemoveFilter = (id: string) => {
        const newSelectedLabelIds = selectedLabelIds.filter((selectedId) => selectedId !== id);

        setSelectedLabelIds(newSelectedLabelIds);
    };

    const filteredLabels = selectedLabelIds
        .map((id) => labels.find((label) => label.id === id))
        .filter(Boolean) as Label[];

    return (
        <DialogTrigger hideArrow type='popover'>
            <PressableElement aria-label='Filter by labels'>
                <Flex
                    gap={'size-40'}
                    wrap={'wrap'}
                    width={'size-3000'}
                    height={'size-400'}
                    alignItems={'center'}
                    UNSAFE_className={classes.filterContainer}
                >
                    {filteredLabels.map((label) => (
                        <FilterChips key={label.id} name={label.name} onClose={() => handleRemoveFilter(label.id)} />
                    ))}

                    {isEmpty(filteredLabels) && <Text UNSAFE_className={classes.searchPlaceholder}>Search labels</Text>}
                </Flex>
            </PressableElement>

            <Dialog width={'size-5000'} UNSAFE_className={classes.dialog} aria-label='Filter media items'>
                <Content>
                    <MultiSelectList
                        name='labels'
                        items={labels}
                        maxHeight='size-2000'
                        selectAllLabel='Toggle all'
                        onSelectionChange={handleSelectionChange}
                        defaultSelectedKeys={new Set(selectedLabelIds)}
                    />
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
