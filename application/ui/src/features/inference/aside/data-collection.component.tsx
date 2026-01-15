// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Divider, Flex, Heading, Slider, Switch, Text } from '@geti/ui';
import { usePatchPipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useIsPipelineConfigured } from 'hooks/use-is-pipeline-configured.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

const DEFAULTS = {
    RATE: 12,
    CONFIDENCE_THRESHOLD: 0.5,
    MIN_SAMPLING_INTERVAL: 2.5,
} as const;

export const DataCollection = () => {
    const projectId = useProjectIdentifier();
    const pipelineQuery = usePipeline();
    const canEditPipeline = useIsPipelineConfigured(pipelineQuery.data);
    const patchPipelineMutation = usePatchPipeline();

    const policies = pipelineQuery.data?.data_collection?.policies ?? [];
    const maxDatasetSize = pipelineQuery.data?.data_collection?.max_dataset_size ?? 500;
    const ratePolicy = policies.find((policy) => policy.type === 'fixed_rate');
    const confidencePolicy = policies.find((policy) => policy.type === 'confidence_threshold');

    const serverRate = ratePolicy?.rate ?? DEFAULTS.RATE;
    const serverConfidenceThreshold = confidencePolicy?.confidence_threshold ?? DEFAULTS.CONFIDENCE_THRESHOLD;

    const [localRate, setLocalRate] = useState(serverRate);
    const [localConfidenceThreshold, setLocalConfidenceThreshold] = useState(serverConfidenceThreshold);

    const updatePolicies = (updates: {
        rateEnabled?: boolean;
        rate?: number;
        confidenceEnabled?: boolean;
        confidenceThreshold?: number;
    }) => {
        const newDataCollectionPolicies = {
            max_dataset_size: maxDatasetSize,
            policies: [
                {
                    type: 'fixed_rate' as const,
                    rate: updates.rate ?? serverRate,
                    enabled: updates.rateEnabled ?? ratePolicy?.enabled ?? false,
                },
                {
                    type: 'confidence_threshold' as const,
                    confidence_threshold: updates.confidenceThreshold ?? serverConfidenceThreshold,
                    min_sampling_interval: confidencePolicy?.min_sampling_interval ?? DEFAULTS.MIN_SAMPLING_INTERVAL,
                    enabled: updates.confidenceEnabled ?? confidencePolicy?.enabled ?? false,
                },
            ],
        };

        patchPipelineMutation.mutate({
            params: { path: { project_id: projectId } },
            body: { data_collection: newDataCollectionPolicies },
        });
    };

    const isDisabled = patchPipelineMutation.isPending || !canEditPipeline;

    return (
        <Flex
            key={`${serverRate}-${serverConfidenceThreshold}`}
            UNSAFE_style={{
                padding: 'var(--spectrum-global-dimension-size-200)',
            }}
            direction={'column'}
        >
            <Flex alignItems='center' gap={'size-100'} marginBottom={'size-300'}>
                <Heading level={4}>Data collection</Heading>
            </Flex>
            <Flex direction={'column'} UNSAFE_style={{ overflow: 'hidden auto' }}>
                <Heading level={3} margin={0}>
                    Capture rate
                </Heading>

                <Text marginY={'size-100'}>Capture frames while the stream is running</Text>

                <Switch
                    isSelected={ratePolicy?.enabled ?? false}
                    onChange={(enabled) => updatePolicies({ rateEnabled: enabled })}
                    isDisabled={isDisabled}
                    marginBottom={'size-200'}
                >
                    Toggle auto capturing
                </Switch>

                <Slider
                    step={0.1}
                    minValue={0}
                    maxValue={60}
                    value={localRate}
                    onChange={setLocalRate}
                    onChangeEnd={(rate) => updatePolicies({ rate })}
                    label='Rate'
                    isDisabled={isDisabled || !ratePolicy?.enabled}
                />

                <Divider marginY={'size-400'} size={'S'} />

                <Heading level={3} margin={0}>
                    Confidence threshold
                </Heading>

                <Text marginY={'size-100'}>Capture frames when confidence is below threshold</Text>

                <Switch
                    isSelected={confidencePolicy?.enabled ?? false}
                    onChange={(enabled) => updatePolicies({ confidenceEnabled: enabled })}
                    isDisabled={isDisabled}
                >
                    Confidence threshold
                </Switch>

                <Slider
                    step={0.01}
                    minValue={0}
                    maxValue={1}
                    value={localConfidenceThreshold}
                    onChange={setLocalConfidenceThreshold}
                    onChangeEnd={(confidenceThreshold) => updatePolicies({ confidenceThreshold })}
                    marginY={'size-200'}
                    label='Threshold'
                    isDisabled={isDisabled || !confidencePolicy?.enabled}
                />
            </Flex>
        </Flex>
    );
};
