// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Heading, Radio, Tooltip, TooltipTrigger, View } from '@geti/ui';
import { clsx } from 'clsx';

import type { SchemaModelArchitecture } from '../../../../api/openapi-spec';
import { InfoTooltip } from '../../../../components/info-tooltip/info-tooltip.component';
import { ModelAttributes } from '../model-attributes/model-attributes.component';
import { Ratings } from '../model-rating/attribute-rating.component';
import { TemplateRating } from '../model-rating/model-rating.component';
import { ModelTag } from '../model-tag/model-tag.component';
import { ModelArchitectureTooltipText } from './model-architecture-tooltip.component';

import classes from './model-type.module.scss';

interface ModelTypeProps {
    selectedModelArchitectureId: string | null;
    modelArchitecture: SchemaModelArchitecture;
}

const RATING_MAP: Record<number, Ratings> = {
    1: 'LOW',
    2: 'MEDIUM',
    3: 'HIGH',
};

export const ModelType = ({ modelArchitecture, selectedModelArchitectureId }: ModelTypeProps) => {
    const { id, name, description, support_status, stats } = modelArchitecture;
    const isSelected = selectedModelArchitectureId === modelArchitecture.id;
    const isActive = support_status === 'active';
    const isObsolete = support_status === 'obsolete';
    const isDeprecated = support_status === 'deprecated';

    return (
        <label
            htmlFor={`${id}-radio`}
            aria-label={`model ${name}`}
            className={clsx(classes.selectableItem, { [classes.selectableItemSelected]: isSelected })}
        >
            <View
                position={'relative'}
                paddingX={'size-175'}
                paddingY={'size-125'}
                borderTopWidth={'thin'}
                borderTopEndRadius={'regular'}
                borderTopStartRadius={'regular'}
                borderTopColor={'gray-200'}
                backgroundColor={'gray-200'}
                UNSAFE_style={{ boxSizing: 'border-box' }}
                UNSAFE_className={clsx({ [classes.selectedHeader]: isSelected })}
            >
                <Flex direction={'column'} width={'100%'} height={'100%'} justifyContent={'center'}>
                    <View marginBottom={'size-50'}>
                        <Flex alignItems={'center'} gap={'size-50'}>
                            <View minWidth={0}>
                                <TooltipTrigger placement={'bottom'}>
                                    <Radio
                                        id={`${id}-radio`}
                                        value={id}
                                        aria-label={name}
                                        UNSAFE_className={classes.radioItem}
                                    >
                                        <Heading
                                            UNSAFE_className={clsx(classes.trainTemplateName, {
                                                [classes.selected]: isSelected,
                                            })}
                                        >
                                            {name}
                                        </Heading>
                                    </Radio>

                                    <Tooltip>{name}</Tooltip>
                                </TooltipTrigger>
                            </View>
                            <InfoTooltip
                                id={`${name.toLocaleLowerCase()}-summary-id`}
                                iconColor={isSelected ? 'var(--energy-blue)' : undefined}
                                tooltipText={
                                    <ModelArchitectureTooltipText
                                        description={description}
                                        isDeprecated={isDeprecated}
                                    />
                                }
                            />
                        </Flex>
                    </View>
                    <Flex alignItems={'center'} gap={'size-100'}>
                        {isActive && <ModelTag name={'Active'} isActive />}

                        <ModelTag name={name} />

                        {isObsolete && <ModelTag name={'Obsolete'} isObsolete />}
                        {isDeprecated && <ModelTag name={'Deprecated'} isDeprecated />}
                    </Flex>
                </Flex>
            </View>

            <View
                flex={1}
                paddingX={'size-250'}
                paddingY={'size-225'}
                borderBottomWidth={'thin'}
                borderBottomEndRadius={'regular'}
                borderBottomStartRadius={'regular'}
                borderBottomColor={'gray-100'}
                minHeight={'size-1000'}
                UNSAFE_className={clsx(classes.selectableItemDescription, {
                    [classes.selectedDescription]: isSelected,
                })}
            >
                <Flex direction={'column'} gap={'size-200'}>
                    <TemplateRating
                        accuracy={RATING_MAP[stats.performance_ratings.accuracy]}
                        trainingTime={RATING_MAP[stats.performance_ratings.training_time]}
                        inferenceSpeed={RATING_MAP[stats.performance_ratings.inference_speed]}
                    />
                    <Divider size={'S'} />
                    <ModelAttributes gigaflops={stats.gigaflops} trainableParameters={stats.trainable_parameters} />
                </Flex>
            </View>
        </label>
    );
};
