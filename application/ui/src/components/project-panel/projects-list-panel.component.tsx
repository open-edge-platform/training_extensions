// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Header,
    Heading,
    PhotoPlaceholder,
    Tag,
    Text,
    View,
} from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useNavigate } from 'react-router';

import { paths } from '../../constants/paths';
import { useProjects } from '../../hooks/api/project.hook';
import { ProjectsList } from './projects-list.component';

import classes from './projects-list.module.scss';

interface SelectedProjectProps {
    name: string;
    id: string | undefined;
    isActive: boolean;
}

const SelectedProjectButton = ({ name, id, isActive }: SelectedProjectProps) => {
    return (
        <Flex alignItems={'center'} gap={'size-100'}>
            <Divider alignSelf={'center'} height={'size-400'} orientation={'vertical'} size={'S'} />

            <ActionButton
                aria-label={`Selected project ${name}`}
                isQuiet
                height={'max-content'}
                staticColor={'white'}
                UNSAFE_className={classes.selectedProjectButton}
            >
                <View margin='size-50'>
                    <PhotoPlaceholder name={name} indicator={id ?? name} height={'size-400'} width={'size-400'} />
                </View>
                <Flex direction={'column'} minWidth={0}>
                    <View paddingStart={'size-50'} width={'100%'} UNSAFE_className={classes.projectName}>
                        <span title={name}>{name}</span>
                    </View>
                    {isActive ? <Tag className={classes.statusTag} text={'Active'} /> : null}
                </Flex>
            </ActionButton>
        </Flex>
    );
};

const AddProjectButton = () => {
    const navigate = useNavigate();

    const addProject = () => {
        navigate(paths.project.new({}));
    };

    return (
        <ActionButton
            isQuiet
            width={'100%'}
            marginStart={'size-100'}
            marginEnd={'size-350'}
            UNSAFE_className={classes.addProjectButton}
            onPress={addProject}
        >
            <AddCircle />
            <Text marginX='size-50'>Create project</Text>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const projectId = useProjectIdentifier();
    const { data } = useProjects();

    const selectedProject = data.find((project) => project.id === projectId);
    const selectedProjectName = selectedProject?.name ?? '';
    const hasActivePipeline = Boolean(selectedProject?.active_pipeline);

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton name={selectedProjectName} id={projectId} isActive={hasActivePipeline} />

            <Dialog width={'size-4600'} UNSAFE_className={classes.dialog}>
                <Header>
                    <Flex
                        direction={'column'}
                        justifyContent={'center'}
                        width={'100%'}
                        alignItems={'center'}
                        UNSAFE_style={{
                            padding: 'var(--spectrum-global-dimension-size-200)',
                        }}
                    >
                        <PhotoPlaceholder
                            name={selectedProjectName}
                            indicator={selectedProject?.id ?? selectedProjectName}
                            height={'size-1000'}
                            width={'size-1000'}
                        />
                        <Heading UNSAFE_style={{ textAlign: 'center' }} level={2} marginBottom={0}>
                            {selectedProjectName}
                        </Heading>
                        {hasActivePipeline ? <Tag text={'Active'} /> : null}
                    </Flex>
                </Header>

                <Divider size={'S'} marginY={'size-200'} />

                <Content>
                    <ProjectsList projects={data} />
                </Content>

                <Divider size={'S'} marginY={'size-200'} />

                <ButtonGroup UNSAFE_className={classes.panelButtons}>
                    <AddProjectButton />
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
