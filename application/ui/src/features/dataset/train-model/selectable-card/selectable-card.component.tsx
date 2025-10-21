// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { ReactNode } from 'react';

import { Flex, View, type DimensionValue, type Responsive } from '@geti/ui';
import { usePress } from 'react-aria';

import classes from './selectable-card.module.scss';

interface SelectableCardProps {
    className?: string;
    isSelected: boolean;
    headerContent: ReactNode;
    descriptionContent: ReactNode;
    minHeight?: Responsive<DimensionValue> | undefined;
    handleOnPress: () => void;
}

export const SelectableCard = ({
    minHeight,
    className = '',
    isSelected,
    headerContent,
    descriptionContent,
    handleOnPress,
}: SelectableCardProps) => {
    const { pressProps } = usePress({
        onPress: () => {
            handleOnPress();
        },
    });

    return (
        <div
            {...pressProps}
            aria-label={isSelected ? 'Selected card' : 'Not selected card'}
            className={[classes.selectableCard, isSelected ? classes.selectableCardSelected : '', className].join(' ')}
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
                minHeight={minHeight ?? 'size-1000'}
                UNSAFE_style={{ boxSizing: 'border-box' }}
                UNSAFE_className={isSelected ? classes.selectedHeader : ''}
            >
                <Flex direction={'column'} width={'100%'} height={'100%'} justifyContent={'center'}>
                    {headerContent}
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
                UNSAFE_className={[
                    classes.selectableCardDescription,
                    isSelected ? classes.selectedDescription : '',
                ].join(' ')}
            >
                {descriptionContent}
            </View>
        </div>
    );
};
