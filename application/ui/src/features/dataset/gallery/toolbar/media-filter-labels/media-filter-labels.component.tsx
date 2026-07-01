// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, Dialog, DialogTrigger, Flex, PressableElement, Text } from '@geti-ui/ui';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { isEmpty } from 'lodash-es';

import { MultiSelectList } from '../../../../../components/multi-select-list/multi-select-list.component';

import classes from './media-filter-labels.module.scss';

const pluralRules = new Intl.PluralRules('en');

export const MediaFilterLabels = () => {
    const labels = useProjectLabels();
    const { selectedLabelIds, setSelectedLabelIds } = useDatasetFiltersSearchParams();

    const handleSelectionChange = (selectedKeys: Set<string> | 'all') => {
        const ids = selectedKeys === 'all' ? labels.map(({ id }) => id) : Array.from(selectedKeys);

        setSelectedLabelIds(ids);
    };

    return (
        <DialogTrigger hideArrow type='popover'>
            <PressableElement>
                <div role='button' aria-label='Filter by labels'>
                    <Flex
                        gap={'size-40'}
                        wrap={'wrap'}
                        width={'size-3000'}
                        height={'size-400'}
                        alignItems={'center'}
                        UNSAFE_className={classes.filterContainer}
                    >
                        {isEmpty(selectedLabelIds) ? (
                            <Text UNSAFE_className={classes.searchPlaceholder}>Search labels</Text>
                        ) : (
                            <Text>{`${selectedLabelIds.length} ${
                                pluralRules.select(selectedLabelIds.length) === 'one' ? 'label' : 'labels'
                            } selected`}</Text>
                        )}
                    </Flex>
                </div>
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
