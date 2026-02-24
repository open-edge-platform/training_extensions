// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import { ActionButton, Flex, Item, Picker, SearchField, Switch, Text, View } from '@geti/ui';
import { DownloadIcon } from '@geti/ui/icons';

import { useAutoScroll } from './hooks/use-auto-scroll.hook';
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
    onDownload?: () => void;
};

const CONNECTION_STATUS_LABEL: Record<ConnectionStatus, string> = {
    connecting: 'Connecting...',
    connected: 'Live',
    disconnected: 'Disconnected',
    error: 'Connection error',
};

export const LogViewer = ({ logs, isStreaming = false, connectionStatus, onDownload }: LogViewerProps) => {
    const [minLevel, setMinLevel] = useState<LogLevel>('INFO');
    const [searchQuery, setSearchQuery] = useState('');

    const filteredLogs = useMemo(() => filterLogs(logs, minLevel, searchQuery), [logs, minLevel, searchQuery]);

    const { scrollRef, autoScroll, setAutoScroll, handleScroll } = useAutoScroll({
        itemCount: filteredLogs.length,
        isDisabled: !isStreaming,
    });

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

                {isStreaming ? (
                    <Switch isSelected={autoScroll} onChange={setAutoScroll} aria-label={'Auto-scroll'}>
                        Auto-scroll
                    </Switch>
                ) : null}

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
                    {onDownload !== undefined ? (
                        <ActionButton
                            isQuiet
                            onPress={onDownload}
                            aria-label={'Download logs'}
                            isDisabled={logs.length === 0}
                        >
                            <DownloadIcon />
                        </ActionButton>
                    ) : null}
                </Flex>
            </Flex>

            <div className={classes.logList} ref={scrollRef} onScroll={handleScroll} role={'list'}>
                {filteredLogs.length > 0 ? (
                    filteredLogs.map((entry, index) => <LogEntry key={index} entry={entry} />)
                ) : (
                    <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                        <Text UNSAFE_className={classes.emptyState}>
                            {logs.length === 0 ? 'No log entries' : 'No matching log entries'}
                        </Text>
                    </Flex>
                )}
            </div>
        </Flex>
    );
};
