// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactComponent as FolderIcon } from '../../../../../assets/icons/folder.svg';
import { ReactComponent as MqttIcon } from '../../../../../assets/icons/mqtt.svg';
import { ReactComponent as RosIcon } from '../../../../../assets/icons/ros.svg';
import { ReactComponent as WebhookIcon } from '../../../../../assets/icons/webhook.svg';

interface SinkIconProps {
    type: 'folder' | 'mqtt' | 'webhook' | 'ros' | 'disconnected';
}

export const SinkIcon = ({ type }: SinkIconProps) => {
    if (type === 'folder') {
        return <FolderIcon />;
    }

    if (type === 'mqtt') {
        return <MqttIcon />;
    }

    if (type === 'ros') {
        return <RosIcon />;
    }

    if (type === 'webhook') {
        return <WebhookIcon />;
    }

    return <></>;
};
