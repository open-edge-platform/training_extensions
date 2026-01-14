// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Text,
    Tooltip,
    TooltipTrigger,
} from '@geti/ui';
import { Close, Hotkeys as HotkeysIcon } from '@geti/ui/icons';

import { HotkeysList } from './hotkeys-list.component';

import styles from './hotkeys.module.scss';

export const Hotkeys = () => {
    return (
        <DialogTrigger type={'popover'} hideArrow placement={'right'}>
            <TooltipTrigger>
                <ActionButton isQuiet aria-label={'Hotkeys'}>
                    <HotkeysIcon />
                </ActionButton>
                <Tooltip>Hotkeys</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog UNSAFE_className={styles.hotkeysDialog}>
                    <Heading>
                        <Flex justifyContent={'space-between'} alignItems={'center'}>
                            <Text>Hotkeys</Text>
                            <ActionButton isQuiet onPress={close} aria-label={'Close hotkeys'}>
                                <Close />
                            </ActionButton>
                        </Flex>
                    </Heading>
                    <Divider size={'S'} />
                    <Content>
                        <HotkeysList />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
