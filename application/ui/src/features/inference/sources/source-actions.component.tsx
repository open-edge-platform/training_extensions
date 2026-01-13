// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, Loading, Text } from '@geti/ui';
import { Back } from '@geti/ui/icons';
import { isEmpty } from 'lodash-es';

import { $api } from '../../../api/client';
import { EditSourceForm } from './edit-source-form.component';
import { SourcesList } from './source-list/source-list.component';
import { SourceOptions } from './source-options';
import { SourceConfig } from './util';

export const SourceActions = () => {
    const [view, setView] = useState<'list' | 'options' | 'edit'>('list');
    const [currentSource, setCurrentSource] = useState<SourceConfig | null>(null);
    const { data: sources = [], isLoading } = $api.useSuspenseQuery('get', '/api/sources');
    const filteredSources = sources.filter((source) => source.source_type !== 'disconnected');

    const handleShowList = () => {
        setView('list');
    };

    const handleAddSource = () => {
        setView('options');
    };

    const handleEditSource = (source: SourceConfig) => {
        setView('edit');
        setCurrentSource(source);
    };

    if (isLoading) {
        return <Loading mode={'inline'} size='M' />;
    }

    if (view === 'edit' && !isEmpty(currentSource)) {
        return <EditSourceForm config={currentSource} onSaved={handleShowList} onBackToList={handleShowList} />;
    }

    if (view === 'list') {
        return <SourcesList sources={filteredSources} onAddSource={handleAddSource} onEditSource={handleEditSource} />;
    }

    return (
        <SourceOptions onSaved={handleShowList} hasHeader={filteredSources.length > 0}>
            <Flex gap={'size-100'} marginBottom={'size-100'} alignItems={'center'} justifyContent={'space-between'}>
                <ActionButton isQuiet onPress={handleShowList}>
                    <Back />
                </ActionButton>

                <Text>Add new input source</Text>
            </Flex>
        </SourceOptions>
    );
};
