// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';

import { TrainingConfiguration } from '../../../../../../../../core/configurable-parameters/services/configuration.interface';
import { isBoolParameter } from '../../../../../../../../core/configurable-parameters/utils';
import {
    getMockedConfigurationParameter,
    getMockedTrainingConfiguration,
} from '../../../../../../../../test-utils/mocked-items-factory/mocked-configuration-parameters';
import { providersRender as render } from '../../../../../../../../test-utils/required-providers-render';
import { DataAugmentation } from './data-augmentation.component';

type DataAugmentationParameters = TrainingConfiguration['datasetPreparation']['augmentation'];

const getToggleEnableParameter = (name: string) => {
    return screen.getByRole('switch', { name: `Toggle ${name}` });
};

const toggleParameter = (name: string) => {
    fireEvent.click(getToggleEnableParameter(name));
};

const getParameter = (name: string) => {
    return screen.getByRole('textbox', { name: `Change ${name}` });
};

describe('DataAugmentation', () => {
    const dataAugmentationParameters = {
        center_crop: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable center crop',
                value: true,
                description: 'Whether to apply center cropping to the image',
                defaultValue: true,
            }),
            getMockedConfigurationParameter({
                key: 'ratio',
                type: 'float',
                name: 'Crop ratio',
                value: 0.6,
                description: 'Ratio of original dimensions to keep when cropping',
                defaultValue: 1,
                maxValue: null,
                minValue: 0,
            }),
        ],
        random_affine: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable random affine',
                value: true,
                description: 'Whether to apply random affine transformations to the image',
                defaultValue: true,
            }),
            getMockedConfigurationParameter({
                key: 'degrees',
                type: 'float',
                name: 'Rotation degrees',
                value: 15,
                description: 'Maximum rotation angle in degrees',
                defaultValue: 0,
                maxValue: null,
                minValue: 0,
            }),
            getMockedConfigurationParameter({
                key: 'translate_x',
                type: 'float',
                name: 'Horizontal translation',
                value: 0,
                description: 'Maximum horizontal translation as a fraction of image width',
                defaultValue: 0,
                maxValue: null,
                minValue: 0,
            }),
            getMockedConfigurationParameter({
                key: 'translate_y',
                type: 'float',
                name: 'Vertical translation',
                value: 0,
                description: 'Maximum vertical translation as a fraction of image height',
                defaultValue: 0,
                maxValue: null,
                minValue: 0,
            }),
            getMockedConfigurationParameter({
                key: 'scale',
                type: 'float',
                name: 'Scale factor',
                value: 1,
                description: 'Scaling factor for the image during affine transformation',
                defaultValue: 1,
                maxValue: null,
                minValue: 0,
            }),
        ],
    } satisfies DataAugmentationParameters;

    const App = (props: { dataAugmentationParameters: DataAugmentationParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(() =>
            getMockedTrainingConfiguration({
                datasetPreparation: {
                    filtering: {},
                    subsetSplit: [],
                    augmentation: props.dataAugmentationParameters,
                },
            })
        );

        const handleUpdateTrainingConfiguration = (
            updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
        ) => {
            setTrainingConfiguration(updateFunction);
        };
        return (
            <DataAugmentation
                parameters={trainingConfiguration?.datasetPreparation.augmentation ?? props.dataAugmentationParameters}
                onUpdateTrainingConfiguration={handleUpdateTrainingConfiguration}
            />
        );
    };

    it('tag displays "Yes" when at least one data augmentation parameter is enabled, "No" otherwise', () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        const enableCenterCrop = dataAugmentationParameters.center_crop[0];
        const enableRandomAffine = dataAugmentationParameters.random_affine[0];

        expect(getToggleEnableParameter(enableCenterCrop.name)).toBeChecked();
        expect(screen.getByLabelText('Data augmentation tag')).toHaveTextContent('Yes');

        toggleParameter(enableCenterCrop.name);
        toggleParameter(enableRandomAffine.name);

        expect(getToggleEnableParameter(enableCenterCrop.name)).not.toBeChecked();
        expect(getToggleEnableParameter(enableRandomAffine.name)).not.toBeChecked();
        expect(screen.getByLabelText('Data augmentation tag')).toHaveTextContent('No');
    });

    it('disables all data augmentation parameters when the "Enable" parameter is false, otherwise are enabled', () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        Object.values(dataAugmentationParameters).forEach((parametersGroup) => {
            const enableParameter = parametersGroup[0];
            expect(getToggleEnableParameter(enableParameter.name)).toBeChecked();

            const restOfParameters = parametersGroup.slice(1);

            restOfParameters.forEach((parameter) => {
                expect(getParameter(parameter.name)).toBeEnabled();
            });

            toggleParameter(enableParameter.name);
            expect(getToggleEnableParameter(enableParameter.name)).not.toBeChecked();

            restOfParameters.forEach((parameter) => {
                expect(getParameter(parameter.name)).toBeDisabled();
            });
        });
    });

    it('updates parameters and resets them to default properly', () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        const parameters = Object.values(dataAugmentationParameters).flat();
        parameters.forEach((parameter) => {
            if (isBoolParameter(parameter)) {
                expect(getToggleEnableParameter(parameter.name)).toBeChecked();

                toggleParameter(parameter.name);

                expect(getToggleEnableParameter(parameter.name)).not.toBeChecked();

                fireEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));
                expect(getToggleEnableParameter(parameter.name)).toBeChecked();
            } else {
                expect(getParameter(parameter.name)).toHaveValue(parameter.value.toString());

                fireEvent.click(screen.getByRole('button', { name: `Increase Change ${parameter.name}` }));

                expect(getParameter(parameter.name)).toHaveValue((Number(parameter.value) + 0.1).toString());

                fireEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));

                expect(getParameter(parameter.name)).toHaveValue(parameter.defaultValue.toString());
            }
        });
    });
});
