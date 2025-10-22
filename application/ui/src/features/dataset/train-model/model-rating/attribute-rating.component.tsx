// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FC } from 'react';

import { Flex, Heading, View } from '@geti/ui';

import classes from './attribute-rating.module.scss';

export type Ratings = 'LOW' | 'MEDIUM' | 'HIGH';

const RateColorPalette = {
    LOW: 'var(--energy-blue-tint2)',
    MEDIUM: 'var(--energy-blue-tint1)',
    HIGH: 'var(--energy-blue)',
    EMPTY: 'var(--spectrum-global-color-gray-500)',
};

const RateColors: Record<Ratings, string[]> = {
    LOW: [RateColorPalette.LOW, RateColorPalette.EMPTY, RateColorPalette.EMPTY],
    MEDIUM: [RateColorPalette.LOW, RateColorPalette.MEDIUM, RateColorPalette.EMPTY],
    HIGH: [RateColorPalette.LOW, RateColorPalette.MEDIUM, RateColorPalette.HIGH],
};
const rateLabels = Object.keys(RateColors);

interface AttributeRatingProps {
    name: string;
    rating: Ratings;
}

export const AttributeRating: FC<AttributeRatingProps> = ({ name, rating }) => {
    return (
        <div aria-label={`Attribute rating for ${name} is ${rating}`} style={{ height: '100%' }}>
            <Flex direction={'column'} gap={'size-100'} justifyContent={'space-between'} height={'100%'}>
                <Heading margin={0} UNSAFE_className={classes.attributeRatingTitle}>
                    {name}
                </Heading>
                <Flex alignItems={'center'} gap={'size-100'}>
                    {RateColors[rating].map((color, idx) => (
                        <View
                            key={`rate-${rateLabels[idx]}`}
                            UNSAFE_className={classes.rate}
                            UNSAFE_style={{ backgroundColor: color }}
                        />
                    ))}
                </Flex>
            </Flex>
        </div>
    );
};
