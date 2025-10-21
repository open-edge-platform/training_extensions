// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { Grid, repeat } from '@geti/ui';

import { AttributeRating, Ratings } from './attribute-rating.component';

interface TemplateRatingProps {
    inferenceSpeed: Ratings;
    trainingTime: Ratings;
    accuracy: Ratings;
}

export const TemplateRating = ({ inferenceSpeed, trainingTime, accuracy }: TemplateRatingProps) => {
    return (
        <Grid columns={repeat(3, '1fr')} justifyContent={'space-evenly'} gap={'size-250'}>
            <AttributeRating name={'Inference speed'} rating={inferenceSpeed} />
            <AttributeRating name={'Training time'} rating={trainingTime} />
            <AttributeRating name={'Accuracy'} rating={accuracy} />
        </Grid>
    );
};
