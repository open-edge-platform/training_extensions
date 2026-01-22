// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { useState } from 'react';

import { fireEvent, screen, within } from '@testing-library/react';

import { TrainingConfiguration } from '../../../../../../../../core/configurable-parameters/services/configuration.interface';
import {
    getMockedConfigurationParameter,
    getMockedTrainingConfiguration,
} from '../../../../../../../../test-utils/mocked-items-factory/mocked-configuration-parameters';
import { providersRender as render } from '../../../../../../../../test-utils/required-providers-render';
import { TrainingSubsets } from './training-subsets.component';

type SubsetsParameters = TrainingConfiguration['datasetPreparation']['subsetSplit'];

const expectSubsetSizes = ({
    trainingSize,
    validationSize,
    testSize,
}: {
    trainingSize: number;
    testSize: number;
    validationSize: number;
}) => {
    expect(screen.getByLabelText('Training subset size')).toHaveTextContent(new RegExp(trainingSize.toString()));
    expect(screen.getByLabelText('Validation subset size')).toHaveTextContent(new RegExp(validationSize.toString()));
    expect(screen.getByLabelText('Test subset size')).toHaveTextContent(new RegExp(testSize.toString()));
};

const expectTrainingSubsetsDistribution = ({
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

describe('TrainingSubsets', () => {
    const subsetsParameters = [
        getMockedConfigurationParameter({
            key: 'training',
            type: 'int',
            name: 'Training percentage',
            value: 70,
            description: 'Percentage of data to use for training',
            defaultValue: 70,
            maxValue: 100,
            minValue: 1,
        }),
        getMockedConfigurationParameter({
            key: 'validation',
            type: 'int',
            name: 'Validation percentage',
            value: 20,
            description: 'Percentage of data to use for validation',
            defaultValue: 20,
            maxValue: 100,
            minValue: 1,
        }),
        getMockedConfigurationParameter({
            key: 'test',
            type: 'int',
            name: 'Test percentage',
            value: 10,
            description: 'Percentage of data to use for testing',
            defaultValue: 10,
            maxValue: 100,
            minValue: 1,
        }),
        getMockedConfigurationParameter({
            key: 'auto_selection',
            type: 'bool',
            name: 'Auto selection',
            value: true,
            description: 'Whether to automatically select data for each subset',
            defaultValue: true,
        }),
        getMockedConfigurationParameter({
            key: 'remixing',
            type: 'bool',
            name: 'Remixing',
            value: false,
            description: 'Whether to remix data between subsets',
            defaultValue: false,
        }),
        getMockedConfigurationParameter({
            key: 'dataset_size',
            type: 'int',
            name: 'Dataset size',
            value: 99,
            description: 'Total size of the dataset (read-only parameter, not configurable by users)',
            defaultValue: 99,
            maxValue: null,
            minValue: 0,
        }),
    ] satisfies SubsetsParameters;

    const [trainingSubset, validationSubset, testSubset] = subsetsParameters;
    const datasetSize = Number(subsetsParameters.at(-1)?.value);

    const App = ({
        subsetParameters,
        hasSupportedModels = false,
    }: {
        subsetParameters: SubsetsParameters;
        hasSupportedModels?: boolean;
    }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(() =>
            getMockedTrainingConfiguration({
                datasetPreparation: {
                    subsetSplit: subsetParameters,
                    filtering: {},
                    augmentation: {},
                },
            })
        );

        const handleUpdateTrainingConfiguration = (
            updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
        ) => {
            setTrainingConfiguration(updateFunction);
        };

        return (
            <TrainingSubsets
                hasSupportedModels={hasSupportedModels}
                subsetsParameters={trainingConfiguration?.datasetPreparation.subsetSplit ?? subsetParameters}
                defaultSubsetParameters={subsetParameters}
                onUpdateTrainingConfiguration={handleUpdateTrainingConfiguration}
            />
        );
    };

    it('displays subsets distribution properly', () => {
        render(<App subsetParameters={subsetsParameters} hasSupportedModels />);

        expectTrainingSubsetsDistribution({
            validationSubset: Number(validationSubset.value),
            testSubset: Number(testSubset.value),
            trainingSubset: Number(trainingSubset.value),
        });

        const validationSize = Math.floor(datasetSize * (Number(validationSubset.value) / 100));
        const testSize = Math.floor(datasetSize * (Number(testSubset.value) / 100));
        const trainingSize = datasetSize - validationSize - testSize;

        expectSubsetSizes({ trainingSize, validationSize, testSize });
    });

    it('updates subsets sizes when changing dataset distribution and resets to default', () => {
        render(<App subsetParameters={subsetsParameters} />);

        for (let i = 0; i < 10; i++) {
            fireEvent.keyDown(screen.getByLabelText('Start range'), { key: 'Left' });

            if (i < 5) {
                fireEvent.keyDown(screen.getByLabelText('End range'), { key: 'Left' });
            }
        }

        const updatedTrainingSubsetDistribution = Number(trainingSubset.value) - 10;
        const updatedValidationSubsetDistribution = Number(validationSubset.value) + 5;
        const updatedTestSubsetDistribution = Number(testSubset.value) + 5;

        const validationSize = Math.floor(datasetSize * (updatedValidationSubsetDistribution / 100));
        const testSize = Math.floor(datasetSize * (updatedTestSubsetDistribution / 100));
        const trainingSize = datasetSize - validationSize - testSize;

        expectTrainingSubsetsDistribution({
            validationSubset: updatedValidationSubsetDistribution,
            testSubset: updatedTestSubsetDistribution,
            trainingSubset: updatedTrainingSubsetDistribution,
        });

        expectSubsetSizes({ trainingSize, validationSize, testSize });

        fireEvent.click(screen.getByRole('button', { name: 'Reset training subsets' }));

        expectTrainingSubsetsDistribution({
            validationSubset: Number(validationSubset.value),
            testSubset: Number(testSubset.value),
            trainingSubset: Number(trainingSubset.value),
        });

        const defaultValidationSize = Math.floor(datasetSize * (Number(validationSubset.value) / 100));
        const defaultTestSize = Math.floor(datasetSize * (Number(testSubset.value) / 100));
        const defaultTrainingSize = datasetSize - defaultTestSize - defaultValidationSize;

        expectSubsetSizes({
            trainingSize: defaultTrainingSize,
            validationSize: defaultValidationSize,
            testSize: defaultTestSize,
        });
    });

    it('shows warning that training subsets is unavailable when there are not enough media items', () => {
        const mockedSubsetsParameters = [
            getMockedConfigurationParameter({
                key: 'training',
                type: 'int',
                name: 'Training percentage',
                value: 70,
                description: 'Percentage of data to use for training',
                defaultValue: 70,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'validation',
                type: 'int',
                name: 'Validation percentage',
                value: 20,
                description: 'Percentage of data to use for validation',
                defaultValue: 20,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'test',
                type: 'int',
                name: 'Test percentage',
                value: 10,
                description: 'Percentage of data to use for testing',
                defaultValue: 10,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'auto_selection',
                type: 'bool',
                name: 'Auto selection',
                value: true,
                description: 'Whether to automatically select data for each subset',
                defaultValue: true,
            }),
            getMockedConfigurationParameter({
                key: 'remixing',
                type: 'bool',
                name: 'Remixing',
                value: false,
                description: 'Whether to remix data between subsets',
                defaultValue: false,
            }),
            getMockedConfigurationParameter({
                key: 'dataset_size',
                type: 'int',
                name: 'Dataset size',
                value: 6,
                description: 'Total size of the dataset (read-only parameter, not configurable by users)',
                defaultValue: 6,
                maxValue: null,
                minValue: 0,
            }),
        ];
        const [_, mockedValidationSubset, mockedTestSubset] = mockedSubsetsParameters;
        const mockedDatasetSize = Number(mockedSubsetsParameters.at(-1)?.value);

        render(<App subsetParameters={mockedSubsetsParameters} />);

        expect(screen.queryByRole('alert')).not.toBeInTheDocument();

        fireEvent.keyDown(screen.getByLabelText('End range'), { key: 'Left' });

        const alert = screen.getByRole('alert');

        expect(alert).toBeInTheDocument();
        expect(within(alert).getByRole('heading')).toHaveTextContent('Invalid training subsets configuration');

        const validationSize = Math.floor(mockedDatasetSize * (Number(mockedValidationSubset.value) / 100));
        const testSize = Math.floor(mockedDatasetSize * (Number(mockedTestSubset.value) / 100));
        const trainingSize = mockedDatasetSize - validationSize - testSize;

        expectSubsetSizes({
            trainingSize,
            validationSize,
            testSize,
        });
    });

    it('shows warning when updated subset distribution requires enabling reshuffle and training from scratch', () => {
        render(<App subsetParameters={subsetsParameters} hasSupportedModels />);

        expect(screen.queryByRole('alert')).not.toBeInTheDocument();

        fireEvent.keyDown(screen.getByLabelText('End range'), { key: 'Left' });

        const alert = screen.getByRole('alert');

        expect(alert).toBeInTheDocument();
        expect(within(alert).getByRole('heading')).toHaveTextContent(
            'Additional configuration change required to apply new training subsets distribution'
        );
    });
});
