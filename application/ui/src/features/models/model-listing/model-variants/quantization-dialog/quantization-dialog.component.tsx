// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Content, Dialog, dimensionValue, Divider, Flex, Heading, Text, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';

import { CalibrationDatasetSizeField, MaxAccuracyDropField } from './quantization-fields.component';

type QuantizationDialogProps = {
    onClose: () => void;
};
export const QuantizationDialog = ({ onClose }: QuantizationDialogProps) => {
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

                        <MaxAccuracyDropField />

                        <CalibrationDatasetSizeField />

                        <Flex gap={'size-100'} alignItems={'center'} marginTop={'size-300'}>
                            <InfoOutline />
                            <Text
                                UNSAFE_style={{
                                    fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                                    color: 'var(--spectrum-global-color-gray-700)',
                                }}
                            >
                                Recommended calibration dataset size: between 200-500 media items
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
