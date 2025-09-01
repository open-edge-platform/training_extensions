// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Divider, Flex, Loading, Switch, Text, Tooltip, TooltipTrigger, useMediaQuery, View } from '@geti/ui';
import { RightClick } from '@geti/ui/icons';
import { isLargeSizeQuery } from '@geti/ui/theme';

import { useSegmentAnything } from './segment-anything-state-provider.component';

const INTERACTIVE_MODE_TOOLTIP = 'With this mode ON, edit preview by placing new positive or negative points. - SHIFT';

const RIGHT_CLICK_MODE_TOOLTIP =
    'With this mode ON, press left-click to place positive points and right-click to place negative points.';

// TODO: replace by actual tool settings
const toolSettings = {
    interactiveMode: false,
    rightClickMode: false,
    maskOpacity: 0.5,
};

export const SecondaryToolbar = () => {
    const isLargeSize = useMediaQuery(isLargeSizeQuery);

    const { points, isLoading, encodingQuery } = useSegmentAnything();

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const handleOnChange = (_value: number) => {
        // TODO: Update tool settings
        // updateToolSettings(ToolType.SegmentAnythingTool, { ...toolSettings, maskOpacity: value });
    };

    const setToolSetting = (_data: unknown) => {
        // TODO: Update tool settings
        // updateToolSettings(ToolType.SegmentAnythingTool, { ...toolSettings, ...data });
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

            {/* <TooltipTrigger placement={'bottom'}>
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
            </TooltipTrigger> */}

            {points.length > 0 && (
                <>
                    <Divider orientation='vertical' size='S' />

                    <Button>Accept</Button>

                    <Button>Reject</Button>
                </>
            )}
        </Flex>
    );
};
