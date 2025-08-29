// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Loading, Switch, Text, Tooltip, TooltipTrigger, useMediaQuery, View } from '@geti/ui';
import { RightClick } from '@geti/ui/icons';
import { isLargeSizeQuery } from '@geti/ui/theme';
import { isEmpty } from 'lodash-es';

import { useSegmentAnything } from './segment-anything-state-provider.component';

const INTERACTIVE_MODE_TOOLTIP = 'With this mode ON, edit preview by placing new positive or negative points. - SHIFT';

const RIGHT_CLICK_MODE_TOOLTIP =
    'With this mode ON, press left-click to place positive points and right-click to place negative points.';

export const SecondaryToolbar = () => {
    const isLargeSize = useMediaQuery(isLargeSizeQuery);

    const { handleCancelAnnotation, handleConfirmAnnotation, result, points, isProcessing, isLoading, encodingQuery } =
        useSegmentAnything();

    const hasResults = !isEmpty(result.shapes) && !isEmpty(points);

    const handleOnChange = (value: number) => {
        updateToolSettings(ToolType.SegmentAnythingTool, { ...toolSettings, maskOpacity: value });
    };

    const setToolSetting = (data: Partial<ToolSettings[ToolType.SegmentAnythingTool]>) => {
        updateToolSettings(ToolType.SegmentAnythingTool, { ...toolSettings, ...data });
    };

    if (isLoading || encodingQuery.data === undefined) {
        return (
            <Flex direction='row' alignItems='center' justifyContent='center' gap='size-125'>
                {isLargeSize && (
                    <>
                        <Text>Auto segmentation</Text>
                        <Divider orientation='vertical' size='S' />
                    </>
                )}

                <Loading mode='inline' size={'S'} />
                <Text>{isLoading ? 'Loading image model' : 'Extracting image features'}</Text>
            </Flex>
        );
    }

    return (
        <Flex direction='row' alignItems='center' justifyContent='center' gap='size-125'>
            {isLargeSize && (
                <>
                    <Text>Auto segmentation</Text>
                    <Divider orientation='vertical' size='S' />
                </>
            )}

            <TooltipTrigger placement={'bottom'}>
                <Switch
                    isSelected={toolSettings.interactiveMode}
                    onChange={(interactiveMode) => setToolSetting({ interactiveMode })}
                    height='100%'
                    aria-label='Interactive mode'
                    isDisabled={toolSettings.interactiveMode === true && points.length > 0}
                >
                    Interactive mode
                </Switch>
                <Tooltip>{INTERACTIVE_MODE_TOOLTIP}</Tooltip>
            </TooltipTrigger>

            <Divider orientation='vertical' size='S' />

            <Flex justifyContent={'center'} alignItems={'center'} marginEnd={'size-200'}>
                <TooltipTrigger placement={'bottom'}>
                    <Switch
                        margin={0}
                        isSelected={toolSettings.rightClickMode}
                        isDisabled={toolSettings.interactiveMode === false}
                        onChange={(val) => setToolSetting({ rightClickMode: val })}
                        height='100%'
                        aria-label='Right-click mode'
                    >
                        <Flex gap={'size-100'} alignContent={'center'} height={'100%'}>
                            Right-click mode
                            <View
                                paddingY={'size-25'}
                                UNSAFE_style={{
                                    color:
                                        toolSettings.rightClickMode && toolSettings.interactiveMode === true
                                            ? 'var(--spectrum-global-color-gray-900)'
                                            : 'var(--spectrum-global-color-gray-600)',
                                }}
                            >
                                <RightClick />
                            </View>
                        </Flex>
                    </Switch>

                    <Tooltip>{RIGHT_CLICK_MODE_TOOLTIP}</Tooltip>
                </TooltipTrigger>
            </Flex>

            <Divider orientation='vertical' size='S' />

            <TooltipTrigger placement={'bottom'}>
                <NumberSliderWithLocalHandler
                    id='mask-opacity'
                    displayText={(value) => `${Math.round(100 * value)}%`}
                    label={'Mask opacity'}
                    ariaLabel='Mask opacity'
                    min={0}
                    max={1}
                    step={0.01}
                    onChange={handleOnChange}
                    value={toolSettings.maskOpacity}
                />
                <Tooltip>Adjust the opacity</Tooltip>
            </TooltipTrigger>

            {points.length > 0 && (
                <>
                    <Divider orientation='vertical' size='S' />

                    <AcceptRejectButtonGroup
                        id={'segment-anything'}
                        isAcceptButtonDisabled={isProcessing || !hasResults}
                        shouldShowButtons={points.length > 0}
                        handleAcceptAnnotation={handleConfirmAnnotation}
                        handleRejectAnnotation={handleCancelAnnotation}
                        acceptDeps={[handleConfirmAnnotation]}
                        rejectDeps={[isProcessing]}
                    />
                </>
            )}
        </Flex>
    );
};
