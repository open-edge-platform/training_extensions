// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, DialogContainer } from '@geti-ui/ui';

type EnablePipelineBlockedDialogProps = {
    isOpen: boolean;
    onClose: () => void;
};

export const EnablePipelineBlockedDialog = ({ isOpen, onClose }: EnablePipelineBlockedDialogProps) => {
    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <AlertDialog
                    title={'Cannot enable pipeline'}
                    primaryActionLabel={'Close'}
                    variant={'warning'}
                    onPrimaryAction={onClose}
                >
                    Make sure you selected a model and source before enabling the pipeline.
                </AlertDialog>
            )}
        </DialogContainer>
    );
};
