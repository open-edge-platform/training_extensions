// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import { ActionButton, Content, ContextualHelp, Flex, Heading, Item, Picker, SearchField, Text, View } from '@geti/ui';
import { ChevronDownLight } from '@geti/ui/icons';

import { useScrollAnchor } from './hooks/use-scroll-anchor.hook';
import { LogEntry } from './log-entry.component';
import { type LogEntry as LogEntryType, type LogLevel } from './log-types';
import { filterLogs } from './log-utils';

import classes from './log-viewer.module.scss';

const LOG_LEVELS: LogLevel[] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

type LogViewerProps = {
    logs: LogEntryType[];
    isStreaming?: boolean;
    connectionStatus?: ConnectionStatus;
};

const CONNECTION_STATUS_LABEL: Record<ConnectionStatus, string> = {
    connecting: 'Connecting...',
    connected: 'Live',
    disconnected: 'Disconnected',
    error: 'Connection error',
};

export const LogViewer = ({ logs, isStreaming = false, connectionStatus }: LogViewerProps) => {
    const [minLevel, setMinLevel] = useState<LogLevel>('INFO');
    const [searchQuery, setSearchQuery] = useState('');

    const filteredLogs = useMemo(() => filterLogs(logs, minLevel, searchQuery), [logs, minLevel, searchQuery]);

    const { anchorRef, isAtBottom, scrollToBottom } = useScrollAnchor();

    return (
        <Flex direction={'column'} height={'100%'} UNSAFE_style={{ overflow: 'hidden' }}>
            <Flex alignItems={'center'} gap={'size-200'} UNSAFE_className={classes.toolbar} flexShrink={0}>
                <Picker
                    label={'Level'}
                    labelPosition={'side'}
                    selectedKey={minLevel}
                    onSelectionChange={(key) => setMinLevel(key as LogLevel)}
                    width={'size-2000'}
                    aria-label={'Minimum log level'}
                    isQuiet
                    contextualHelp={
                        <ContextualHelp variant={'info'}>
                            <Heading>Minimum log level</Heading>
                            <Content>
                                <Text>
                                    Shows log entries at the selected level and above. For example, selecting WARNING
                                    shows WARNING, ERROR, and CRITICAL entries, hiding DEBUG and INFO.
                                </Text>
                            </Content>
                        </ContextualHelp>
                    }
                >
                    {LOG_LEVELS.map((level) => (
                        <Item key={level}>{level}</Item>
                    ))}
                </Picker>

                <SearchField
                    value={searchQuery}
                    onChange={setSearchQuery}
                    placeholder={'Search logs...'}
                    aria-label={'Search logs'}
                    width={'size-3000'}
                    isQuiet
                />

                <Flex alignItems={'center'} gap={'size-100'} marginStart={'auto'}>
                    {connectionStatus !== undefined ? (
                        <Flex alignItems={'center'} gap={'size-75'}>
                            <View
                                UNSAFE_className={`${classes.statusDot} ${classes[`statusDot--${connectionStatus}`]}`}
                            />
                            <Text UNSAFE_className={classes.statusText}>
                                {CONNECTION_STATUS_LABEL[connectionStatus]}
                            </Text>
                        </Flex>
                    ) : null}
                    <Text UNSAFE_className={classes.logCount}>
                        {filteredLogs.length} / {logs.length} entries
                    </Text>
                </Flex>
            </Flex>

            <div className={classes.logListContainer}>
                <div className={classes.logList} role={'list'}>
                    {filteredLogs.length > 0 ? (
                        filteredLogs.map((entry, index) => <LogEntry key={index} entry={entry} />)
                    ) : (
                        <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                            <Text UNSAFE_className={classes.emptyState}>
                                {logs.length === 0 ? 'No log entries' : 'No matching log entries'}
                            </Text>
                        </Flex>
                    )}
                    <div ref={anchorRef} className={classes.scrollAnchor} />
                </div>

                {isStreaming && !isAtBottom ? (
                    <ActionButton
                        onPress={scrollToBottom}
                        UNSAFE_className={classes.scrollToBottom}
                        aria-label={'Scroll to bottom'}
                    >
                        <ChevronDownLight />
                        <Text>Scroll to bottom</Text>
                    </ActionButton>
                ) : null}
            </div>
        </Flex>
    );
};
