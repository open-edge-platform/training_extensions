// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { Divider, Flex, Heading, Slider, Switch, Text } from '@geti/ui';
import { useIsPipelineConfigured } from 'hooks/use-is-pipeline-configured.hook';
import { $api } from 'src/api/client';
import { components } from 'src/api/openapi-spec';
import { useProjectIdentifier } from 'src/hooks/use-project-identifier.hook';

type FixedRateDataCollectionPolicy = components['schemas']['FixedRateDataCollectionPolicy'];
type ConfidenceThresholdDataCollectionPolicy = components['schemas']['ConfidenceThresholdDataCollectionPolicy'];

export const DataCollection = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const canEditPipeline = useIsPipelineConfigured(pipelineQuery.data);

    const patchPipelineMutation = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/pipeline']],
        },
    });

    const isAutoCapturingEnabled = pipelineQuery.data?.data_collection_policies[0]?.enabled ?? false;
    const defaultRate = 12;
    const serverRate =
        (pipelineQuery.data?.data_collection_policies[0] as FixedRateDataCollectionPolicy)?.rate ?? defaultRate;

    const confidencePolicy = pipelineQuery.data?.data_collection_policies[1] as
        | ConfidenceThresholdDataCollectionPolicy
        | undefined;
    const isConfidenceThresholdEnabled = confidencePolicy?.enabled ?? false;
    const defaultConfidenceThreshold = 0.5;
    const serverConfidenceThreshold = confidencePolicy?.confidence_threshold ?? defaultConfidenceThreshold;
    const defaultMinSamplingInterval = 2.5;
    const serverMinSamplingInterval = confidencePolicy?.min_sampling_interval ?? defaultMinSamplingInterval;

    const [localRate, setLocalRate] = useState(serverRate);
    const [localConfidenceThreshold, setLocalConfidenceThreshold] = useState(serverConfidenceThreshold);

    useEffect(() => {
        setLocalRate(serverRate);
    }, [serverRate]);

    useEffect(() => {
        setLocalConfidenceThreshold(serverConfidenceThreshold);
    }, [serverConfidenceThreshold]);

    const toggleAutoCapturing = (isEnabled: boolean) => {
        const confidencePolicyBody = confidencePolicy
            ? [
                  {
                      type: 'confidence_threshold' as const,
                      confidence_threshold: confidencePolicy.confidence_threshold,
                      min_sampling_interval: confidencePolicy.min_sampling_interval,
                      enabled: confidencePolicy.enabled,
                  },
              ]
            : [];

        patchPipelineMutation.mutate({
            params: { path: { project_id: projectId } },
            body: {
                data_collection_policies: [
                    { type: 'fixed_rate' as const, rate: defaultRate, enabled: isEnabled },
                    ...confidencePolicyBody,
                ],
            },
        });
    };

    const updateRate = (value: number) => {
        const confidencePolicyBody = confidencePolicy
            ? [
                  {
                      type: 'confidence_threshold' as const,
                      confidence_threshold: confidencePolicy.confidence_threshold,
                      min_sampling_interval: confidencePolicy.min_sampling_interval,
                      enabled: confidencePolicy.enabled,
                  },
              ]
            : [];

        patchPipelineMutation.mutate({
            params: { path: { project_id: projectId } },
            body: {
                data_collection_policies: [
                    { type: 'fixed_rate' as const, rate: value, enabled: isAutoCapturingEnabled },
                    ...confidencePolicyBody,
                ],
            },
        });
    };

    const toggleConfidenceThreshold = (isEnabled: boolean) => {
        patchPipelineMutation.mutate({
            params: { path: { project_id: projectId } },
            body: {
                data_collection_policies: [
                    { type: 'fixed_rate' as const, rate: serverRate, enabled: isAutoCapturingEnabled },
                    {
                        type: 'confidence_threshold' as const,
                        confidence_threshold: serverConfidenceThreshold,
                        min_sampling_interval: serverMinSamplingInterval,
                        enabled: isEnabled,
                    },
                ],
            },
        });
    };

    const updateConfidenceThreshold = (value: number) => {
        patchPipelineMutation.mutate({
            params: { path: { project_id: projectId } },
            body: {
                data_collection_policies: [
                    { type: 'fixed_rate' as const, rate: serverRate, enabled: isAutoCapturingEnabled },
                    {
                        type: 'confidence_threshold' as const,
                        confidence_threshold: value,
                        min_sampling_interval: serverMinSamplingInterval,
                        enabled: isConfidenceThresholdEnabled,
                    },
                ],
            },
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
                <Heading level={3} margin={0}>
                    Capture rate
                </Heading>

                <Text marginY={'size-100'}>Capture frames while the stream is running</Text>

                <Switch
                    isSelected={isAutoCapturingEnabled}
                    onChange={toggleAutoCapturing}
                    isDisabled={patchPipelineMutation.isPending || !canEditPipeline}
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
                    onChangeEnd={updateRate}
                    label='Rate'
                    isDisabled={patchPipelineMutation.isPending || !canEditPipeline || !isAutoCapturingEnabled}
                />

                <Divider marginY={'size-400'} size={'S'} />

                <Heading level={3} margin={0}>
                    Confidence threshold
                </Heading>

                <Text marginY={'size-100'}>Capture frames when confidence is below threshold</Text>

                <Switch
                    isSelected={isConfidenceThresholdEnabled}
                    onChange={toggleConfidenceThreshold}
                    isDisabled={patchPipelineMutation.isPending || !canEditPipeline}
                >
                    Confidence threshold
                </Switch>

                <Slider
                    step={0.01}
                    minValue={0}
                    maxValue={1}
                    value={localConfidenceThreshold}
                    onChange={setLocalConfidenceThreshold}
                    onChangeEnd={updateConfidenceThreshold}
                    marginY={'size-200'}
                    label='Threshold'
                    isDisabled={patchPipelineMutation.isPending || !canEditPipeline || !isConfidenceThresholdEnabled}
                />
            </Flex>
        </Flex>
    );
};
