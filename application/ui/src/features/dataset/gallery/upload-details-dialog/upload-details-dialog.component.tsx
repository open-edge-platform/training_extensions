// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Cell,
    Column,
    Content,
    Dialog,
    DialogContainer,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Loading,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { AcceptCircle, CrossCircle, Pending } from '@geti-ui/ui/icons';

import { formatBytes, pluralizeItems } from '../../../../shared/util';
import { useMediaUploadContext } from '../../providers/media-upload-provider.component';
import { computeSummary, type UploadFileItem, type UploadItemStatus } from '../../providers/media-upload-reducer';

import classes from './upload-details-dialog.module.scss';

const STATUS_LABEL: Record<UploadItemStatus, string> = {
    queued: 'Queued',
    uploading: 'Uploading',
    uploaded: 'Uploaded',
    failed: 'Failed',
};

const StatusIcon = ({ status }: { status: UploadItemStatus }): ReactNode => {
    switch (status) {
        case 'queued':
            return <Pending aria-label={'Queued'} size={'S'} />;
        case 'uploading':
            return <Loading mode={'inline'} size={'S'} />;
        case 'uploaded':
            return (
                <AcceptCircle aria-label={'Uploaded'} width={16} height={16} style={{ fill: 'var(--brand-moss)' }} />
            );
        case 'failed':
            return (
                <CrossCircle
                    aria-label={'Failed'}
                    width={16}
                    height={16}
                    style={{ fill: 'var(--brand-coral-cobalt)' }}
                />
            );
    }
};

const StatusCell = ({ item }: { item: UploadFileItem }) => {
    const statusContent = (
        <Flex alignItems={'center'} gap={'size-100'}>
            <StatusIcon status={item.status} />
            <Text>{STATUS_LABEL[item.status]}</Text>
        </Flex>
    );

    if (item.status === 'failed' && item.errorMessage) {
        return (
            <Flex alignItems={'center'} gap={'size-100'}>
                {statusContent}
                <DialogTrigger type={'popover'}>
                    <ActionButton isQuiet aria-label={'Error details'} UNSAFE_className={classes.error}>
                        Error
                    </ActionButton>
                    <Dialog>
                        <Heading>Upload error</Heading>
                        <Divider />
                        <Content>
                            <Text>{item.errorMessage}</Text>
                        </Content>
                    </Dialog>
                </DialogTrigger>
            </Flex>
        );
    }

    return statusContent;
};

const buildSubheader = (total: number, succeeded: number, failed: number, isUploading: boolean): string => {
    if (isUploading) {
        const parts = [`${succeeded} uploaded`, failed > 0 ? `${failed} failed` : null].filter(Boolean).join(', ');

        return `Uploading ${total} ${pluralizeItems(total)} — ${parts}`;
    }

    if (failed === 0) return `Uploaded ${succeeded} ${pluralizeItems(succeeded)}`;
    if (succeeded === 0) return `Failed to upload ${failed} ${pluralizeItems(failed)}`;

    return `Uploaded ${succeeded} ${pluralizeItems(succeeded)}, ${failed} failed`;
};

const UploadDetailsDialogContent = ({ onClose }: { onClose: () => void }) => {
    const { state } = useMediaUploadContext();
    const summary = computeSummary(state.items, state.isUploading);
    const items = state.items;

    const subheader = buildSubheader(summary.total, summary.succeeded, summary.failed, summary.isUploading);

    return (
        <Dialog size={'L'}>
            <Heading>Upload details</Heading>
            <Divider />
            <Content>
                <Flex direction={'column'} gap={'size-200'}>
                    <Text>{subheader}</Text>
                    <TableView
                        aria-label={'Upload details'}
                        overflowMode={'truncate'}
                        density={'compact'}
                        maxHeight={'60vh'}
                        isQuiet
                    >
                        <TableHeader>
                            <Column isRowHeader>FILENAME</Column>
                            <Column width={160}>STATUS</Column>
                            <Column width={120} align={'end'}>
                                SIZE
                            </Column>
                        </TableHeader>
                        <TableBody items={items}>
                            {(item) => (
                                <Row key={item.id}>
                                    <Cell>
                                        <TooltipTrigger>
                                            <Text>{item.name}</Text>
                                            <Tooltip>{item.name}</Tooltip>
                                        </TooltipTrigger>
                                    </Cell>
                                    <Cell>
                                        <StatusCell item={item} />
                                    </Cell>
                                    <Cell>{formatBytes(item.size)}</Cell>
                                </Row>
                            )}
                        </TableBody>
                    </TableView>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant={'primary'} onPress={onClose}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export const UploadDetailsDialog = () => {
    const { state, dispatch } = useMediaUploadContext();
    const close = () => dispatch({ type: 'CLOSE_DIALOG' });

    return (
        <DialogContainer onDismiss={close}>
            {state.isDetailsDialogOpen && <UploadDetailsDialogContent onClose={close} />}
        </DialogContainer>
    );
};
