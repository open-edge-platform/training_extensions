// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Grid, Heading, ToggleButton, View } from '@geti/ui';

import { ToolSelectionBar } from '../../../components/tool-selection-bar/tool-selection-bar.component';
import { AnnotatorCanvas } from '../../annotator/annotator-canvas';
import { response } from '../mock-response';

type Item = (typeof response.items)[number];

export const MediaPreview = ({ mediaItem, close }: { mediaItem: Item; close: () => void }) => {
    const [isFocussed, setIsFocussed] = useState(false);

    return (
        <Dialog UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>
            <Divider />
            <Content>
                <View height='100%'>
                    <Grid
                        areas={['toolbar canvas aside', 'toolbar canvas aside', 'toolbar footer aside']}
                        width={'100%'}
                        height='100%'
                        columns={'auto 1fr auto'}
                        rows={'auto 1fr auto'}
                        UNSAFE_style={{
                            border: 'thin solid var(--spectrum-global-color-gray-50)',
                            backgroundColor: 'var(--spectrum-global-color-gray-50)',
                        }}
                    >
                        <View gridArea={'toolbar'} backgroundColor={'gray-100'} padding={'size-100'}>
                            <ToolSelectionBar />
                        </View>

                        <View gridArea={'canvas'} backgroundColor={'gray-50'}>
                            <AnnotatorCanvas mediaItem={mediaItem} isFocussed={isFocussed} />
                        </View>

                        <View gridArea={'aside'} backgroundColor={'gray-50'}>
                            <div>Aside</div>
                        </View>

                        <View
                            gridArea={'footer'}
                            padding={'size-100'}
                            backgroundColor={'gray-100'}
                            UNSAFE_style={{ textAlign: 'right' }}
                        >
                            <ButtonGroup>
                                <ToggleButton isEmphasized isSelected={isFocussed} onChange={setIsFocussed}>
                                    Focus
                                </ToggleButton>
                                <Button variant='secondary' onPress={close}>
                                    Close
                                </Button>
                            </ButtonGroup>
                        </View>
                    </Grid>
                </View>
            </Content>
        </Dialog>
    );
};
