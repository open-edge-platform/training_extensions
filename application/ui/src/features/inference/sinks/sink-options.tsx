// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactComponent as FolderIcon } from '../../../assets/icons/folder.svg';
import { ReactComponent as MqttIcon } from '../../../assets/icons/mqtt.svg';
import { ReactComponent as WebhookIcon } from '../../../assets/icons/webhook.svg';
import { DisclosureGroup } from '../sources/disclosure-group.component';
import { LocalFolder } from './local-folder/local-folder.component';
import { Mqtt } from './mqtt/mqtt.component';
import { Webhook } from './webhook/webhook.component';

const inputs = [
    { label: 'Folder', value: 'folder', content: <LocalFolder />, icon: <FolderIcon width={'24px'} /> },
    { label: 'Webhook', value: 'webhook', content: <Webhook />, icon: <WebhookIcon width={'24px'} /> },
    { label: 'MQTT', value: 'mqtt', content: <Mqtt />, icon: <MqttIcon width={'24px'} /> },
];

export const SinkOptions = () => {
    return <DisclosureGroup items={inputs} defaultActiveInput={null} />;
};
