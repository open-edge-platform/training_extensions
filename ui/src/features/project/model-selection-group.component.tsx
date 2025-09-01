// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex, Heading, Image, Radio, RadioGroup, Text, View } from '@geti/ui';

import thumbnailUrl from '../../assets/mocked-project-thumbnail.png';

import classes from './model-selection.module.scss';

type Model = {
    id: string;
    imageSrc: string;
    title: string;
    description: string;
    verb: string;
    value: string;
};
const MODELS: Model[] = [
    {
        id: 'detection_model',
        imageSrc: thumbnailUrl,
        title: 'Object Detection',
        description: 'Identify and locate objects in your images',
        verb: 'detect',
        value: 'detection',
    },
    {
        id: 'segmentation_model',
        imageSrc: thumbnailUrl,
        title: 'Image Segmentation',
        description: 'Detect and outline specific regions or shapes',
        verb: 'segment',
        value: 'segmentation',
    },
    {
        id: 'classification_model',
        imageSrc: thumbnailUrl,
        title: 'Image Classification',
        description: 'Categorize entire images based on their content',
        verb: 'classify',
        value: 'classification',
    },
];

type ModelOptionProps = {
    model: Model;
    onPress: () => void;
};
const ModelOption = ({ model, onPress }: ModelOptionProps) => {
    return (
        <div onClick={onPress} className={classes.option} aria-label={`Model option: ${model.title}`}>
            <View maxWidth={'344px'}>
                <Image height={'size-3000'} width={'size-3600'} src={model.imageSrc} alt={model.title} />
            </View>

            <View padding={'size-300'} backgroundColor={'gray-100'}>
                <Flex justifyContent={'space-between'} alignItems={'center'}>
                    <Heading level={2} UNSAFE_className={classes.title}>
                        {model.title}
                    </Heading>
                    <Radio aria-label={model.value} value={model.value} />
                </Flex>

                <Text UNSAFE_className={classes.description}>{model.description}</Text>
            </View>
        </div>
    );
};

export const ModelSelectionGroup = () => {
    const [selectedOption, setSelectedOption] = useState(MODELS[0]);

    return (
        <Flex direction={'column'} gap={'size-300'} alignItems={'center'}>
            <RadioGroup
                aria-label='Model selection'
                value={selectedOption.value}
                onChange={(value: string) => {
                    const selectedModel = MODELS.find((model) => model.value === value);

                    if (selectedModel) setSelectedOption(selectedModel);
                }}
            >
                <Flex justifyContent={'center'} gap={'size-300'}>
                    {MODELS.map((model) => (
                        <ModelOption
                            key={model.value}
                            model={model}
                            onPress={() => {
                                setSelectedOption(model);
                            }}
                        />
                    ))}
                </Flex>
            </RadioGroup>

            <Flex>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>
                    {`What objects should the model learn to ${selectedOption.verb}?`}
                </Text>
            </Flex>
        </Flex>
    );
};
