// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { Divider, Flex, Heading, Slider, Switch, Text } from '@geti/ui';
import { $api } from 'src/api/client';
import { components } from 'src/api/openapi-spec';
import { useProjectIdentifier } from 'src/hooks/use-project-identifier.hook';

type FixedRateDataCollectionPolicy = components['schemas']['FixedRateDataCollectionPolicy'];

export const DataCollection = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const patchPipelineMutation = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/pipeline']],
        },
    });

    const isAutoCapturingEnabled = pipelineQuery.data?.data_collection_policies[0]?.enabled ?? false;
    const defaultRate = 12;
    const serverRate =
        (pipelineQuery.data?.data_collection_policies[0] as FixedRateDataCollectionPolicy)?.rate ?? defaultRate;

    // TODO: add confidence_threshold slider

    const [localRate, setLocalRate] = useState(serverRate);

    useEffect(() => {
        setLocalRate(serverRate);
    }, [serverRate]);

    const toggleAutoCapturing = (isEnabled: boolean) => {
        patchPipelineMutation.mutate({
            params: { path: { project_id: projectId } },
            body: { data_collection_policies: [{ rate: defaultRate, enabled: isEnabled }] },
        });
    };

    const updateRate = (value: number) => {
        patchPipelineMutation.mutate({
            params: { path: { project_id: projectId } },
            body: { data_collection_policies: [{ rate: value, enabled: isAutoCapturingEnabled }] },
        });
    };

    return (
        <Flex
            UNSAFE_style={{
                padding: 'var(--spectrum-global-dimension-size-200)',
            }}
            direction={'column'}
        >
            <Flex alignItems='center' gap={'size-100'} marginBottom={'size-300'}>
                <Heading level={4}>Data collection</Heading>
            </Flex>
            <Flex direction={'column'} UNSAFE_style={{ overflow: 'hidden auto' }}>
                <Switch
                    isSelected={isAutoCapturingEnabled}
                    onChange={toggleAutoCapturing}
                    isDisabled={patchPipelineMutation.isPending}
                    marginBottom={'size-200'}
                >
                    Toggle auto capturing
                </Switch>

                <Text>Some description about this stuff</Text>

                <Divider marginY={'size-400'} size={'S'} />

                <Heading level={4} margin={0}>
                    Capture rate
                </Heading>

                <Slider
                    step={0.1}
                    minValue={0}
                    maxValue={60}
                    value={localRate}
                    onChange={setLocalRate}
                    onChangeEnd={updateRate}
                    marginY={'size-200'}
                    label='Rate'
                    isDisabled={patchPipelineMutation.isPending}
                />
            </Flex>
        </Flex>
    );
};
