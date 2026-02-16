// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    Content,
    Dialog,
    Divider,
    Flex,
    Header,
    Heading,
    Loading,
    Text,
    useDialogContainer,
} from '@geti/ui';
import { CloseSemiBold } from '@geti/ui/icons';

import { useModelLogs } from './hooks/use-model-logs.hook';
import { useStreamJobLogs } from './hooks/use-stream-job-logs.hook';
import { LogViewer } from './log-viewer.component';

import classes from './training-logs-dialog.module.scss';

type TrainingLogsDialogProps = {
    jobId?: string;
    modelId?: string;
};

const ActiveJobLogs = ({ jobId }: { jobId: string }) => {
    const { logs, connectionStatus } = useStreamJobLogs(jobId);

    return <LogViewer logs={logs} isStreaming connectionStatus={connectionStatus} />;
};

const HistoricalModelLogs = ({ modelId }: { modelId: string }) => {
    const { data: logs, isPending, isError, error } = useModelLogs(modelId);

    if (isPending) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                <Loading size={'M'} />
            </Flex>
        );
    }

    if (isError) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                <Text UNSAFE_className={classes.errorText}>
                    Failed to load logs: {error?.message ?? 'Unknown error'}
                </Text>
            </Flex>
        );
    }

    return <LogViewer logs={logs ?? []} />;
};

export const TrainingLogsDialog = ({ jobId, modelId }: TrainingLogsDialogProps) => {
    const title = jobId ? 'Training Logs (Live)' : 'Training Logs';
    const dialogContainer = useDialogContainer();

    return (
        <Dialog aria-label={'Training logs'} UNSAFE_className={classes.dialog}>
            <Heading>{title}</Heading>
            <Header>
                <ActionButton
                    isQuiet
                    onPress={dialogContainer.dismiss}
                    aria-label={'Close dialog'}
                    UNSAFE_className={classes.closeButton}
                >
                    <CloseSemiBold width={14} height={14} />
                </ActionButton>
            </Header>
            <Divider />
            <Content UNSAFE_className={classes.contentArea}>
                {jobId && <ActiveJobLogs jobId={jobId} />}
                {!jobId && modelId && <HistoricalModelLogs modelId={modelId} />}
                {!jobId && !modelId && (
                    <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                        <Text UNSAFE_className={classes.errorText}>No job or model specified</Text>
                    </Flex>
                )}
            </Content>
        </Dialog>
    );
};
