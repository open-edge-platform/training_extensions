// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex, Heading, Image, Radio, RadioGroup, Text, View } from '@geti/ui';

import classes from './model-selection.module.scss';

const Models = [
    {
        id: 'detection_model',
        imageSrc: 'https://images.unsplash.com/photo-1461749280684-dccba630e2f6',
        title: 'Object Detection',
        description: 'Identify and locate objects in your images',
        value: 'detection',
    },
    {
        id: 'segmentation_model',
        imageSrc: 'https://images.unsplash.com/photo-1461749280684-dccba630e2f6',
        title: 'Image Segmentation',
        description: 'Detect and outline specific regions or shapes',
        value: 'segmentation',
    },
    {
        id: 'classification_model',
        imageSrc: 'https://images.unsplash.com/photo-1461749280684-dccba630e2f6',
        title: 'Image Classification',
        description: 'Categorize entire images based on their content',
        value: 'classification',
    },
];

type ModelOptionProps = {
    imageSrc: string;
    title: string;
    description: string;
    value: string;
    onPress: () => void;
};
const ModelOption = ({ imageSrc, title, description, value, onPress }: ModelOptionProps) => {
    return (
        <div onClick={onPress} className={classes.option} aria-label={`Model option: ${title}`}>
            <View maxWidth={'344px'}>
                <Image height={'size-3000'} src={imageSrc} alt={title} />
            </View>

            <View padding={'size-300'} backgroundColor={'gray-100'}>
                <Flex justifyContent={'space-between'} alignItems={'center'}>
                    <Heading level={2} UNSAFE_className={classes.title}>
                        {title}
                    </Heading>
                    <Radio aria-label={value} value={value} onFocus={onPress} />
                </Flex>

                <Text UNSAFE_className={classes.description}>{description}</Text>
            </View>
        </div>
    );
};

export const ModelSelectionGroup = () => {
    const [selectedOption, setSelectedOption] = useState(Models[0].value);

    return (
        <RadioGroup aria-label='Model selection' value={selectedOption} onChange={setSelectedOption}>
            <Flex justifyContent={'center'} gap={'size-300'}>
                {Models.map((model) => (
                    <ModelOption
                        key={model.value}
                        imageSrc={model.imageSrc}
                        title={model.title}
                        description={model.description}
                        value={model.value}
                        onPress={() => setSelectedOption(model.value)}
                    />
                ))}
            </Flex>
        </RadioGroup>
    );
};
