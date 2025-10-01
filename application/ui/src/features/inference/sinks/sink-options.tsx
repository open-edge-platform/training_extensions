// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../api/client';
import { ReactComponent as FolderIcon } from '../../../assets/icons/folder.svg';
import { ReactComponent as MqttIcon } from '../../../assets/icons/mqtt.svg';
import { ReactComponent as WebhookIcon } from '../../../assets/icons/webhook.svg';
import { useProjectIdentifier } from '../../../hooks/use-project-identifier.hook';
import { DisclosureGroup } from '../sources/disclosure-group.component';
import { LocalFolder } from './local-folder/local-folder.component';
import { Mqtt } from './mqtt/mqtt.component';
import { getLocalFolderData, getMqttData, getWebhookData } from './utils';
import { Webhook } from './webhook/webhook.component';

export const SinkOptions = () => {
    const projectId = useProjectIdentifier();

    const sinksQuery = $api.useSuspenseQuery('get', '/api/sinks');
    const pipeline = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const sinks = sinksQuery.data ?? [];

    const inputs = [
        {
            label: 'Folder',
            value: 'folder',
            content: <LocalFolder config={getLocalFolderData(sinks)} />,
            icon: <FolderIcon width={'24px'} />,
        },
        {
            label: 'Webhook',
            value: 'webhook',
            content: <Webhook config={getWebhookData(sinks)} />,
            icon: <WebhookIcon width={'24px'} />,
        },
        {
            label: 'MQTT',
            value: 'mqtt',
            content: <Mqtt config={getMqttData(sinks)} />,
            icon: <MqttIcon width={'24px'} />,
        },
    ];

    return <DisclosureGroup items={inputs} defaultActiveInput={pipeline.data?.sink?.sink_type ?? null} />;
};
