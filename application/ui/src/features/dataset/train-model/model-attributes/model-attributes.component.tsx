// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, Heading, repeat } from '@geti/ui';

import classes from './model-attributes.module.scss';

interface ModelAttributesProps {
    gigaflops: number;
    trainableParameters: number;
}
interface ModelAttributeProps {
    value: string;
    title: string;
    gridArea: string;
}

const ModelAttribute = ({ title, value, gridArea }: ModelAttributeProps) => {
    return (
        <>
            <Heading margin={0} UNSAFE_className={classes.attributeTitle} gridArea={`${gridArea}-title`}>
                {title}
            </Heading>
            <span aria-label={title} style={{ gridArea: `${gridArea}-attribute` }}>
                {value}
            </span>
        </>
    );
};

export const ModelAttributes = ({ gigaflops, trainableParameters }: ModelAttributesProps) => {
    return (
        <Grid
            columns={repeat(2, 'max-content')}
            gap={'size-200'}
            areas={['model-size-title complexity-title', 'model-size-attribute complexity-attribute']}
        >
            <ModelAttribute gridArea={'model-size'} title={'Model size'} value={`${trainableParameters} M`} />
            <ModelAttribute gridArea={'complexity'} title={'Complexity'} value={`${gigaflops} GFlops`} />
        </Grid>
    );
};
