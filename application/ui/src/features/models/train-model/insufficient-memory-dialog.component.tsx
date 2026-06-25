// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Content, DialogContainer, Divider, Flex, Heading, Text } from '@geti/ui';

import { InsufficientMemoryDetail } from './insufficient-memory';

type InsufficientMemoryDialogProps = {
    detail: InsufficientMemoryDetail | null;
    onClose: () => void;
    onSelectModel: (modelArchitectureId: string) => void;
};

const formatGb = (megabytes: number): string => `${(megabytes / 1024).toFixed(1)} GB`;

export const InsufficientMemoryDialog = ({ detail, onClose, onSelectModel }: InsufficientMemoryDialogProps) => {
    const topRecommendation = detail?.recommended_models.at(0) ?? null;

    const hasRecommendation = topRecommendation !== null;

    return (
        <DialogContainer onDismiss={onClose}>
            {detail !== null && (
                <AlertDialog
                    title={'Not enough memory to train this model'}
                    variant={'warning'}
                    primaryActionLabel={hasRecommendation ? `Use ${topRecommendation.name}` : 'Close'}
                    onPrimaryAction={() => {
                        if (topRecommendation !== null) {
                            onSelectModel(topRecommendation.id);
                        }
                        onClose();
                    }}
                    cancelLabel={hasRecommendation ? 'Cancel' : undefined}
                    onCancel={onClose}
                >
                    <Flex direction={'column'} gap={'size-150'}>
                        <Text>
                            Training <b>{detail.model_architecture_name}</b> is estimated to need about{' '}
                            {formatGb(detail.estimated_memory_mb)} on {detail.device}, but only about{' '}
                            {formatGb(detail.usable_memory_mb)} is usable (of{' '}
                            {formatGb(detail.available_memory_mb)} total). Starting training would most likely fail
                            with an out-of-memory error.
                        </Text>

                        {detail.recommended_models.length > 0 ? (
                            <>
                                <Divider size={'S'} />
                                <Heading level={4} margin={0}>
                                    Models that should fit
                                </Heading>
                                <Flex direction={'column'} gap={'size-100'}>
                                    {detail.recommended_models.map((model) => (
                                        <Flex
                                            key={model.id}
                                            justifyContent={'space-between'}
                                            alignItems={'center'}
                                            gap={'size-200'}
                                        >
                                            <Content>{model.name}</Content>
                                            <Text>~{formatGb(model.estimated_memory_mb)}</Text>
                                        </Flex>
                                    ))}
                                </Flex>
                            </>
                        ) : (
                            <Text>
                                No lighter model architecture is expected to fit on this device. Consider training on a
                                machine with more memory.
                            </Text>
                        )}
                    </Flex>
                </AlertDialog>
            )}
        </DialogContainer>
    );
};
