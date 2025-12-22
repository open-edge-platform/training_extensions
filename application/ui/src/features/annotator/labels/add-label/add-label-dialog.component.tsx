// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, Dialog } from '@geti/ui';
import { Label as LabelType } from 'src/constants/shared-types';

import { CreateLabelForm } from './create-label-form.component';

interface AddLabelDialogProps {
    closeDialog: () => void;
    existingLabels: LabelType[];
}

export const AddLabelDialog = ({ existingLabels, closeDialog }: AddLabelDialogProps) => {
    return (
        <Dialog>
            <Content>
                <CreateLabelForm onClose={closeDialog} existingLabels={existingLabels} />
            </Content>
        </Dialog>
    );
};
