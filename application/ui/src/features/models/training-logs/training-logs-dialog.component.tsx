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

import { useDownloadLogs } from './hooks/use-download-logs.hook';
import { useModelLogs } from './hooks/use-model-logs.hook';
import { useStreamJobLogs } from './hooks/use-stream-job-logs.hook';
import { LogViewer } from './log-viewer.component';

import classes from './training-logs-dialog.module.scss';

type TrainingLogsDialogProps = {
    jobId?: string;
    modelId?: string;
    modelName?: string;
    trainingDate?: string;
};

const ActiveJobLogs = ({ jobId }: { jobId: string }) => {
    const { logs, connectionStatus } = useStreamJobLogs(jobId);

    return <LogViewer logs={logs} isStreaming connectionStatus={connectionStatus} />;
};

type HistoricalModelLogsProps = {
    modelId: string;
    modelName?: string;
    trainingDate?: string;
};

const HistoricalModelLogs = ({ modelId, modelName, trainingDate }: HistoricalModelLogsProps) => {
    const { data: logs, isPending, isError, error } = useModelLogs(modelId);
    const { downloadLogs } = useDownloadLogs({ modelId, modelName, trainingDate });

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

    return <LogViewer logs={logs ?? []} onDownload={downloadLogs} />;
};

export const TrainingLogsDialog = ({ jobId, modelId, modelName, trainingDate }: TrainingLogsDialogProps) => {
    const dialogContainer = useDialogContainer();

    return (
        <Dialog aria-label={'Training logs'} UNSAFE_className={classes.dialog}>
            <Heading>Training Logs</Heading>
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
                {!jobId && modelId && (
                    <HistoricalModelLogs modelId={modelId} modelName={modelName} trainingDate={trainingDate} />
                )}
                {!jobId && !modelId && (
                    <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                        <Text UNSAFE_className={classes.errorText}>No job or model specified</Text>
                    </Flex>
                )}
            </Content>
        </Dialog>
    );
};
