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
    onSaved: () => void;
    onBackToList: () => void;
}

export const EditSinkForm = ({ config, onSaved, onBackToList }: EditSinkFormProps) => {
    if (config.sink_type === 'folder') {
        return (
            <EditSink
                config={config}
                onSaved={onSaved}
                onBackToList={onBackToList}
                componentFields={(state) => <LocalFolder defaultState={state} />}
                bodyFormatter={localFolderBodyFormatter}
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
            />
        );
    }

    return <></>;
};
