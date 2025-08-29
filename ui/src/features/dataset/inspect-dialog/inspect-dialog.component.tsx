// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Grid, Heading, ToggleButton, View } from '@geti/ui';

import { ImageAnnotations } from '../../annotator/image-annotation';
import { response } from '../mock-response';

type Item = (typeof response.items)[number];

export const InspectDialog = ({ mediaItem, close }: { mediaItem: Item; close: () => void }) => {
    const [isFocussed, setIsFocussed] = useState(false);

    return (
        <Dialog size='L' UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>
            <Divider />
            <Content>
                <View height='100%'>
                    <Grid
                        areas={['toolbar aside', 'canvas aside', 'footer aside']}
                        width={'100%'}
                        height='100%'
                        UNSAFE_style={{
                            border: 'thin solid var(--spectrum-global-color-gray-50)',
                            boxSizing: 'border-box',
                            backgroundColor: 'var(--spectrum-global-color-gray-50)',
                            gridTemplateRows: 'auto 1fr auto',
                            gridTemplateColumns: '1fr auto',
                        }}
                    >
                        <View gridArea={'toolbar'} backgroundColor={'gray-100'} padding={'size-100'}>
                            <Flex justifyContent={'end'}>
                                <ButtonGroup>
                                    <ToggleButton isEmphasized isSelected={isFocussed} onChange={setIsFocussed}>
                                        Focus
                                    </ToggleButton>
                                </ButtonGroup>
                            </Flex>
                        </View>

                        <View gridArea={'canvas'} backgroundColor={'gray-50'}>
                            <ImageAnnotations mediaItem={mediaItem} isFocussed={isFocussed} />
                        </View>

                        <View
                            gridArea={'footer'}
                            padding={'size-100'}
                            backgroundColor={'gray-100'}
                            UNSAFE_style={{ textAlign: 'right' }}
                        >
                            <ButtonGroup>
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
