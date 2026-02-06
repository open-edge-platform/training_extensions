// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Form,
    Heading,
    TextField,
    toast,
} from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { usePatchProject } from '../../../../hooks/api/project.hook';

interface EditProjectNameDialogProps {
    onClose: () => void;
    isOpen: boolean;
    projectId: string;
    projectName: string;
}

const MAX_NUMBER_OF_CHARACTERS_OF_PROJECT_NAME = 100;

export const EditProjectNameDialog = ({ onClose, isOpen, projectId, projectName }: EditProjectNameDialogProps) => {
    const patchProjectMutation = usePatchProject();
    const [newProjectName, setNewProjectName] = useState(projectName);

    const isNameUnchanged = newProjectName.trim() === projectName;
    const isSaveButtonDisabled = isEmpty(newProjectName) || isNameUnchanged || patchProjectMutation.isPending;

    const editProjectName = async (newName: string) => {
        patchProjectMutation.mutate(
            {
                params: { path: { project_id: projectId } },
                body: { name: newName },
            },
            {
                onSuccess: () => {
                    onClose();
                    toast({ type: 'success', message: 'Project updated successfully' });
                },
            }
        );
    };

    const handleEditProjectName = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (isSaveButtonDisabled) {
            return;
        }

        await editProjectName(newProjectName);
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <Dialog>
                    <Heading>Edit project name</Heading>
                    <Divider />
                    <Content>
                        <Form onSubmit={handleEditProjectName}>
                            <TextField
                                maxLength={MAX_NUMBER_OF_CHARACTERS_OF_PROJECT_NAME}
                                //eslint-disable-next-line jsx-a11y/no-autofocus
                                autoFocus
                                value={newProjectName}
                                onChange={setNewProjectName}
                                width='100%'
                                id={'edit-project-name-field-id'}
                                aria-label={'Edit project name field'}
                                isReadOnly={patchProjectMutation.isPending}
                            />
                            <ButtonGroup align={'end'} marginTop={'size-350'}>
                                <Button
                                    variant='secondary'
                                    onPress={onClose}
                                    isDisabled={patchProjectMutation.isPending}
                                >
                                    Cancel
                                </Button>
                                <Button
                                    type='submit'
                                    variant='accent'
                                    isDisabled={isSaveButtonDisabled}
                                    isPending={patchProjectMutation.isPending}
                                >
                                    Save
                                </Button>
                            </ButtonGroup>
                        </Form>
                    </Content>
                </Dialog>
            )}
        </DialogContainer>
    );
};
