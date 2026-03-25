// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    ContextualHelp,
    Dialog,
    dimensionValue,
    Divider,
    Flex,
    Grid,
    Heading,
    Text,
    View,
} from '@geti/ui';
import { InfoOutline, Refresh } from '@geti/ui/icons';

import { NumberParameterField } from '../../../train-model/advanced-settings/components/number-parameter-field.component';

const DEFAULT_QUANTIZATION_PARAMETERS = {
    accuracyDrop: 3.0,
    calibrationSize: 200,
};

type QuantizationFieldLayoutProps = {
    children: React.ReactNode;
    onReset: () => void;
};
const QuantizationFieldLayout = ({ children, onReset }: QuantizationFieldLayoutProps) => (
    <Grid columns={['1fr', '.1fr', 'size-3400', '1fr', '.2fr']} alignItems={'center'} gap={'size-200'}>
        {children}

        <ActionButton isQuiet aria-label={'Reset to default'} onPress={onReset}>
            <Refresh />
        </ActionButton>
    </Grid>
);

type QuantizationDialogProps = {
    onClose: () => void;
};
export const QuantizationDialog = ({ onClose }: QuantizationDialogProps) => {
    const [accuracyDrop, setAccuracyDrop] = useState<number>(DEFAULT_QUANTIZATION_PARAMETERS.accuracyDrop);
    const [calibrationSize, setCalibrationSize] = useState<number>(DEFAULT_QUANTIZATION_PARAMETERS.calibrationSize);

    const model = 'my model';

    return (
        <Dialog width={'100%'}>
            <Heading>Quantization</Heading>

            <Divider size={'S'} />

            <Content>
                <View padding={'size-300'} backgroundColor={'gray-50'} height={'100%'}>
                    <View padding={'size-300'} backgroundColor={'gray-75'} height={'100%'}>
                        <Heading UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }} level={4}>
                            Quantize Model {model} to INT8
                        </Heading>

                        <Divider size={'S'} marginY={'size-200'} />

                        <QuantizationFieldLayout
                            onReset={() => {
                                setAccuracyDrop(DEFAULT_QUANTIZATION_PARAMETERS.accuracyDrop);
                            }}
                        >
                            <Text>Max accuracy drop</Text>
                            <ContextualHelp>
                                <Content>Maximum allowed drop in validation accuracy</Content>
                            </ContextualHelp>
                            <NumberParameterField
                                name='Max accuracy drop'
                                value={accuracyDrop}
                                minValue={0.1}
                                maxValue={10.0}
                                type={'float'}
                                onChange={setAccuracyDrop}
                                step={0.1}
                            />
                            <Checkbox aria-label='No maximum'>No maximum</Checkbox>
                        </QuantizationFieldLayout>

                        <QuantizationFieldLayout
                            onReset={() => {
                                setCalibrationSize(DEFAULT_QUANTIZATION_PARAMETERS.calibrationSize);
                            }}
                        >
                            <Text>Max calibration size</Text>
                            <ContextualHelp>
                                <Content>
                                    Calibration samples will be randomly selected within the validation set
                                </Content>
                            </ContextualHelp>
                            <NumberParameterField
                                name='Max calibration size'
                                value={calibrationSize}
                                minValue={1}
                                maxValue={1000}
                                type={'int'}
                                onChange={setCalibrationSize}
                                step={1}
                            />
                            <Checkbox aria-label='Use full dataset'>Use full dataset</Checkbox>
                        </QuantizationFieldLayout>

                        <Flex gap={'size-100'} alignItems={'center'} marginTop={'size-300'}>
                            <InfoOutline />
                            <Text
                                UNSAFE_style={{
                                    fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                                    color: 'var(--spectrum-global-color-gray-700)',
                                }}
                            >
                                Recommended calibration dataset size: Between 200-500 media items
                            </Text>
                        </Flex>
                    </View>
                </View>
            </Content>

            <ButtonGroup>
                <Button
                    variant={'secondary'}
                    onPress={onClose}
                    UNSAFE_style={{ paddingTop: dimensionValue('size-75') }}
                >
                    Cancel
                </Button>
                <Button variant={'primary'}>Start quantization</Button>
            </ButtonGroup>
        </Dialog>
    );
};
