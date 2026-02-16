// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { LOG_LEVEL_COLORS, type LogEntry as LogEntryType } from './log-types';

import classes from './log-entry.module.scss';

type LogEntryProps = {
    entry: LogEntryType;
};

const formatTimestamp = (timestamp: number): string => {
    if (!timestamp) {
        return '';
    }

    const date = new Date(timestamp * 1000);

    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

const formatSource = (name: string, func: string, line: number): string => {
    if (!name && !func) {
        return '';
    }

    const parts = [name || '', func || ''];

    if (line > 0) {
        parts.push(String(line));
    }

    return parts.filter(Boolean).join(':');
};

export const LogEntry = ({ entry }: LogEntryProps) => {
    const { record } = entry;
    const levelColor = LOG_LEVEL_COLORS[record.level.name] ?? LOG_LEVEL_COLORS.INFO;
    const timestamp = formatTimestamp(record.time.timestamp);
    const source = formatSource(record.name, record.function, record.line);

    return (
        <div className={classes.logEntry} role={'listitem'}>
            {timestamp ? <span className={classes.timestamp}>{timestamp}</span> : null}
            <span className={classes.level} style={{ color: levelColor }}>
                {record.level.name}
            </span>
            {source ? <span className={classes.source}>{source}</span> : null}
            <span className={classes.message}>{record.message.trim()}</span>
        </div>
    );
};
