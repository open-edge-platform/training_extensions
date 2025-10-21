// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { Divider, Flex, Heading, Radio, RadioGroup, Tooltip, TooltipTrigger, View } from '@geti/ui';
import { clsx } from 'clsx';
import { SchemaModelArchitecture } from 'src/api/openapi-spec';
import { InfoTooltip } from 'src/components/info-tooltip/info-tooltip.component';

import { ModelAttributes } from '../model-attributes/model-attributes.component';
import { Ratings } from '../model-rating/attribute-rating.component';
import { TemplateRating } from '../model-rating/model-rating.component';
import { ModelTag } from '../model-tag/model-tag.component';
import { SelectableCard } from '../selectable-card/selectable-card.component';
import { ModelArchitectureTooltipText } from './model-architecture-tooltip.component';

import classes from './model-type.module.scss';

interface ModelTypeProps {
    selectedModelArchitectureId: string | null;
    activeModelTemplateId: string | null;
    modelArchitecture: SchemaModelArchitecture;
    onChangeSelectedTemplateId: (modelTemplateId: string | null) => void;
}

const RATING_MAP: Record<number, Ratings> = {
    1: 'LOW',
    2: 'MEDIUM',
    3: 'HIGH',
};

export const ModelType = ({
    modelArchitecture,
    selectedModelArchitectureId,
    activeModelTemplateId,
    onChangeSelectedTemplateId,
}: ModelTypeProps) => {
    const { id, name, description, support_status, stats } = modelArchitecture;
    const isSelected = selectedModelArchitectureId === modelArchitecture.id;
    const isActive = support_status === 'active';
    const isObsolete = support_status === 'obsolete';
    const isDeprecated = support_status === 'deprecated';
    console.log('modelArchitecture', modelArchitecture);
    stats.performance_ratings;

    const handlePress = () => {
        onChangeSelectedTemplateId(isSelected ? null : id);
    };

    return (
        <SelectableCard
            isSelected={isSelected}
            handleOnPress={handlePress}
            headerContent={
                <>
                    <View marginBottom={'size-50'}>
                        <RadioGroup
                            isEmphasized
                            minWidth={0}
                            aria-label={`Select ${name}`}
                            onChange={handlePress}
                            value={selectedModelArchitectureId}
                            UNSAFE_className={classes.radioGroup}
                        >
                            <Flex alignItems={'center'} gap={'size-50'}>
                                <View minWidth={0}>
                                    <TooltipTrigger placement={'bottom'}>
                                        <Radio value={id} aria-label={name}>
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
                        </RadioGroup>
                    </View>
                    <Flex alignItems={'center'} gap={'size-100'}>
                        {isActive && <ModelTag name={'Active'} isActive />}

                        <ModelTag name={name} />

                        {isObsolete && <ModelTag name={'Obsolete'} isObsolete />}
                        {isDeprecated && <ModelTag name={'Deprecated'} isDeprecated />}
                    </Flex>
                </>
            }
            descriptionContent={
                <Flex direction={'column'} gap={'size-200'}>
                    <TemplateRating
                        accuracy={RATING_MAP[stats.performance_ratings.accuracy]}
                        trainingTime={RATING_MAP[stats.performance_ratings.training_time]}
                        inferenceSpeed={RATING_MAP[stats.performance_ratings.inference_speed]}
                    />
                    <Divider size={'S'} />
                    <ModelAttributes gigaflops={stats.gigaflops} trainableParameters={stats.trainable_parameters} />
                </Flex>
            }
        />
    );
};
