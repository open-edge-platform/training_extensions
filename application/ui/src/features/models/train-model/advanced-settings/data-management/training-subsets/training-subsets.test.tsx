// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../../api/utils';
import { TrainingConfiguration } from '../../../../../../constants/shared-types';
import { server } from '../../../../../../msw-node-setup';
import { distributeByLargestRemainder } from '../../../../utils';
import { TrainingSubsets } from './training-subsets.component';
import { getSubsetSplitParameters, SubsetSplitParameters } from './utils';

const expectSubsetSizes = async ({
    trainingSize,
    validationSize,
    testSize,
}: {
    trainingSize: number;
    testSize: number;
    validationSize: number;
}) => {
    await waitFor(() => {
        expect(screen.getByLabelText('Training subset size')).toHaveTextContent(new RegExp(trainingSize.toString()));
        expect(screen.getByLabelText('Validation subset size')).toHaveTextContent(
            new RegExp(validationSize.toString())
        );
        expect(screen.getByLabelText('Test subset size')).toHaveTextContent(new RegExp(testSize.toString()));
    });
};

const expectTotalSizes = async ({
    unassignedSize,
    assignedSize,
    datasetSize,
}: {
    unassignedSize: number;
    datasetSize: number;
    assignedSize: number;
}) => {
    await waitFor(() => {
        expect(screen.getByLabelText('Total dataset samples')).toHaveTextContent(new RegExp(datasetSize.toString()));
        expect(screen.getByLabelText('Total assigned samples')).toHaveTextContent(new RegExp(assignedSize.toString()));
        expect(screen.getByLabelText('Total size')).toHaveTextContent(unassignedSize.toString());
        expect(screen.getByLabelText('Total unassigned samples')).toHaveTextContent(
            new RegExp(unassignedSize.toString())
        );
    });
};

const expectResultSizes = ({
    trainingResultSize,
    validationResultSize,
    testingResultSize,
}: {
    trainingResultSize: number;
    validationResultSize: number;
    testingResultSize: number;
}) => {
    expect(screen.getByLabelText('Training result size')).toHaveTextContent(new RegExp(trainingResultSize.toString()));
    expect(screen.getByLabelText('Validation result size')).toHaveTextContent(
        new RegExp(validationResultSize.toString())
    );
    expect(screen.getByLabelText('Test result size')).toHaveTextContent(new RegExp(testingResultSize.toString()));
};

const expectTrainingSubsetsDistributionProportion = ({
    trainingSubset,
    validationSubset,
    testSubset,
}: {
    trainingSubset: number;
    validationSubset: number;
    testSubset: number;
}) => {
    expect(screen.getByLabelText('Training subsets tag')).toHaveTextContent(
        `${trainingSubset}/${validationSubset}/${testSubset}%`
    );
    expect(screen.getByLabelText('Training subsets distribution')).toHaveTextContent(
        `${trainingSubset}/${validationSubset}/${testSubset}%`
    );
};

const expectTrainingSubsetsDistribution = async ({
    trainingSize,
    validationSize,
    testSize,
    unassignedSize,
    assignedSize,
    datasetSize,
    trainingSubset,
    validationSubset,
    testSubset,
}: {
    unassignedSize: number;
    trainingSize: number;
    validationSize: number;
    testSize: number;
    assignedSize: number;
    datasetSize: number;
    trainingSubset: number;
    validationSubset: number;
    testSubset: number;
}) => {
    const [newTrainingSize, newValidationSize, newTestingSize] = distributeByLargestRemainder(
        [trainingSubset, validationSubset, testSubset],
        unassignedSize
    );

    expectTrainingSubsetsDistributionProportion({
        validationSubset,
        testSubset,
        trainingSubset,
    });

    await expectSubsetSizes({
        trainingSize: newTrainingSize,
        validationSize: newValidationSize,
        testSize: newTestingSize,
    });

    const trainingResultSize = newTrainingSize + trainingSize;
    const validationResultSize = newValidationSize + validationSize;
    const testingResultSize = newTestingSize + testSize;

    await expectTotalSizes({
        assignedSize,
        unassignedSize,
        datasetSize,
    });

    expectResultSizes({
        trainingResultSize,
        validationResultSize,
        testingResultSize,
    });
};

const mockSubsetsNetworkRequest = ({
    trainingSize,
    validationSize,
    unassignedSize,
    testSize,
}: {
    testSize: number;
    trainingSize: number;
    validationSize: number;
    unassignedSize: number;
}) => {
    server.use(
        http.get('/api/projects/{project_id}/dataset/items', ({ query }) => {
            const subset = query.get('subset');

            if (subset === 'testing') {
                return HttpResponse.json({
                    items: [],
                    pagination: {
                        offset: 0,
                        limit: 1,
                        count: testSize,
                        total: testSize,
                    },
                });
            } else if (subset === 'training') {
                return HttpResponse.json({
                    items: [],
                    pagination: {
                        offset: 0,
                        limit: 1,
                        count: trainingSize,
                        total: trainingSize,
                    },
                });
            } else if (subset === 'validation') {
                return HttpResponse.json({
                    items: [],
                    pagination: {
                        offset: 0,
                        limit: 1,
                        count: validationSize,
                        total: validationSize,
                    },
                });
            } else if (subset === 'unassigned') {
                return HttpResponse.json({
                    items: [],
                    pagination: {
                        offset: 0,
                        limit: 1,
                        count: unassignedSize,
                        total: unassignedSize,
                    },
                });
            }
        })
    );
};

describe('TrainingSubsets', () => {
    const subsetsParameters: SubsetSplitParameters = [
        getMockedConfigurationParameter({
            key: 'training',
            value_type: 'int',
            name: 'Training percentage',
            value: 70,
            description: 'Percentage of data to use for training',
            default_value: 70,
            max_value: 100,
            min_value: 1,
        }),
        getMockedConfigurationParameter({
            key: 'validation',
            value_type: 'int',
            name: 'Validation percentage',
            value: 20,
            description: 'Percentage of data to use for validation',
            default_value: 20,
            max_value: 100,
            min_value: 1,
        }),
        getMockedConfigurationParameter({
            key: 'test',
            value_type: 'int',
            name: 'Test percentage',
            value: 10,
            description: 'Percentage of data to use for testing',
            default_value: 10,
            max_value: 100,
            min_value: 1,
        }),
    ];

    const [trainingSubset, validationSubset, testSubset] = subsetsParameters;

    const App = ({ subsetParameters }: { subsetParameters: SubsetSplitParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>({
            parameters: [
                getMockedConfigurationParameterGroup({
                    key: 'dataset_preparation',
                    parameters: [
                        getMockedConfigurationParameterGroup({
                            key: 'subset_split',
                            parameters: subsetParameters,
                        }),
                    ],
                }),
            ],
        });

        return (
            <TrainingSubsets
                subsetsParameters={
                    getSubsetSplitParameters(trainingConfiguration ?? { parameters: [] }) ?? subsetsParameters
                }
                defaultSubsetParameters={subsetParameters}
                onTrainingConfigurationChange={setTrainingConfiguration}
            />
        );
    };

    it('displays subsets distribution properly', async () => {
        const trainingSize = 10;
        const validationSize = 10;
        const testSize = 10;
        const unassignedSize = 30;
        const assignedSize = trainingSize + validationSize + testSize;
        const datasetSize = assignedSize + unassignedSize;

        mockSubsetsNetworkRequest({
            trainingSize,
            validationSize,
            unassignedSize,
            testSize,
        });

        render(<App subsetParameters={subsetsParameters} />);

        await expectTrainingSubsetsDistribution({
            trainingSize,
            validationSize,
            unassignedSize,
            datasetSize,
            assignedSize,
            testSize,
            trainingSubset: trainingSubset.value,
            validationSubset: validationSubset.value,
            testSubset: testSubset.value,
        });
    });

    it('updates subsets sizes when changing dataset distribution and resets to default', async () => {
        const trainingSize = 10;
        const validationSize = 10;
        const testSize = 10;
        const unassignedSize = 30;
        const assignedSize = trainingSize + validationSize + testSize;
        const datasetSize = assignedSize + unassignedSize;

        mockSubsetsNetworkRequest({
            trainingSize,
            validationSize,
            unassignedSize,
            testSize,
        });

        render(<App subsetParameters={subsetsParameters} />);

        for (let i = 0; i < 10; i++) {
            fireEvent.keyDown(screen.getByLabelText('Start range'), { key: 'Left' });

            if (i < 5) {
                fireEvent.keyDown(screen.getByLabelText('End range'), { key: 'Left' });
            }
        }

        const updatedTrainingSubsetDistribution = trainingSubset.value - 10;
        const updatedValidationSubsetDistribution = validationSubset.value + 5;
        const updatedTestSubsetDistribution = testSubset.value + 5;

        await expectTrainingSubsetsDistribution({
            trainingSize,
            validationSize,
            testSize,
            assignedSize,
            datasetSize,
            unassignedSize,
            trainingSubset: updatedTrainingSubsetDistribution,
            validationSubset: updatedValidationSubsetDistribution,
            testSubset: updatedTestSubsetDistribution,
        });

        fireEvent.click(screen.getByRole('button', { name: 'Reset training subsets' }));

        expectTrainingSubsetsDistributionProportion({
            validationSubset: validationSubset.value,
            testSubset: testSubset.value,
            trainingSubset: trainingSubset.value,
        });

        await expectTrainingSubsetsDistribution({
            trainingSize,
            validationSize,
            testSize,
            assignedSize,
            datasetSize,
            unassignedSize,
            trainingSubset: trainingSubset.value,
            validationSubset: validationSubset.value,
            testSubset: testSubset.value,
        });
    });

    it('shows warning that training subsets is unavailable when there are not enough media items', async () => {
        const trainingSize = 10;
        const validationSize = 10;
        const testSize = 0;
        const unassignedSize = 0;
        const assignedSize = trainingSize + validationSize + testSize;
        const datasetSize = assignedSize + unassignedSize;

        mockSubsetsNetworkRequest({
            trainingSize,
            validationSize,
            unassignedSize,
            testSize,
        });

        render(<App subsetParameters={subsetsParameters} />);

        expect(screen.queryByRole('alert')).not.toBeInTheDocument();

        await expectTrainingSubsetsDistribution({
            trainingSize,
            validationSize,
            testSize,
            assignedSize,
            datasetSize,
            unassignedSize,
            trainingSubset: trainingSubset.value,
            validationSubset: validationSubset.value,
            testSubset: testSubset.value,
        });

        fireEvent.keyDown(screen.getByLabelText('End range'), { key: 'Left' });

        const alert = screen.getByRole('alert');

        expect(alert).toBeInTheDocument();
        expect(within(alert).getByRole('heading')).toHaveTextContent('Invalid training subsets configuration');
    });

    it('45/28/27 split on 5 unassigned items distributes as Training=2, Validation=2, Test=1', async () => {
        const trainingSize = 0;
        const validationSize = 0;
        const testSize = 0;
        const unassignedSize = 5;

        mockSubsetsNetworkRequest({ trainingSize, validationSize, unassignedSize, testSize });

        const regressionSubsetParameters: SubsetSplitParameters = [
            getMockedConfigurationParameter({
                key: 'training',
                value_type: 'int',
                name: 'Training percentage',
                value: 45,
                description: 'Percentage of data to use for training',
                default_value: 45,
                max_value: 100,
                min_value: 1,
            }),
            getMockedConfigurationParameter({
                key: 'validation',
                value_type: 'int',
                name: 'Validation percentage',
                value: 28,
                description: 'Percentage of data to use for validation',
                default_value: 28,
                max_value: 100,
                min_value: 1,
            }),
            getMockedConfigurationParameter({
                key: 'test',
                value_type: 'int',
                name: 'Test percentage',
                value: 27,
                description: 'Percentage of data to use for testing',
                default_value: 27,
                max_value: 100,
                min_value: 1,
            }),
        ];

        render(<App subsetParameters={regressionSubsetParameters} />);

        await expectSubsetSizes({ trainingSize: 2, validationSize: 2, testSize: 1 });
    });
});
