// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { EditSink } from './edit-sink/edit-sink.component';
import { LocalFolder } from './local-folder/local-folder.component';
import { localFolderBodyFormatter } from './local-folder/utils';
import { Mqtt } from './mqtt/mqtt.component';
import { mqttBodyFormatter } from './mqtt/utils';
import { SinkConfig } from './utils';
import { webhookBodyFormatter } from './webhook/utils';
import { Webhook } from './webhook/webhook.component';

interface EditSinkFormProps {
    config: SinkConfig;
    connectedSinkId: string | undefined;
    onSaved: () => void;
    onBackToList: () => void;
}

export const EditSinkForm = ({ config, connectedSinkId, onSaved, onBackToList }: EditSinkFormProps) => {
    if (config.sink_type === 'folder') {
        return (
            <EditSink
                config={config}
                onSaved={onSaved}
                onBackToList={onBackToList}
                componentFields={(state) => <LocalFolder defaultState={state} />}
                bodyFormatter={localFolderBodyFormatter}
                isConnected={connectedSinkId === config.id}
            />
        );
    }

    if (config.sink_type === 'webhook') {
        return (
            <EditSink
                config={config}
                onSaved={onSaved}
                onBackToList={onBackToList}
                componentFields={(state) => <Webhook defaultState={state} />}
                bodyFormatter={webhookBodyFormatter}
                isConnected={connectedSinkId === config.id}
            />
        );
    }

    if (config.sink_type === 'mqtt') {
        return (
            <EditSink
                config={config}
                onSaved={onSaved}
                onBackToList={onBackToList}
                componentFields={(state) => <Mqtt defaultState={state} />}
                bodyFormatter={mqttBodyFormatter}
                isConnected={connectedSinkId === config.id}
            />
        );
    }

    return <></>;
};
