// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { NetworkFixture } from '@msw/playwright';
import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedJob } from 'mocks/mock-job';
import { getMockedModel, getMockedModelArchitecture } from 'mocks/mock-model';
import { HttpResponse } from 'msw';

import {
    NumberEnumConfigurableParameter,
    TrainingConfigurationRequestPayload,
    TrainingRequestPayload,
} from '../../src/constants/shared-types';
import { findGroupByKey } from '../../src/features/models/model-listing/model-training-parameters/utils';
import {
    isInputSizeHeightParameter,
    isInputSizeWidthParameter,
} from '../../src/features/models/train-model/advanced-settings/training/learning-parameters/utils';
import { deepReplaceParameters } from '../../src/features/models/train-model/advanced-settings/utils';
import { getTrainingConfigurationUpdatePayload } from '../../src/features/models/train-model/hooks/utils';
import { expect, http, test } from '../fixtures';
import { MOCKED_MODEL_TRAINING_CONFIGURATION, MOCKED_TRAINING_CONFIGURATION } from './mocks';

const mockedModelArchitectures = [
    getMockedModelArchitecture({ id: 'Object_Detection_SSD', name: 'Object_Detection_SSD' }),
    getMockedModelArchitecture({
        id: 'Custom_Object_Detection_Gen3_ATSS',
        name: 'Custom_Object_Detection_Gen3_ATSS',
    }),
];

const mockedModelRevisions = [
    getMockedModel({
        id: 'model-rev-1',
        name: 'ATSS Revision 1',
        architecture: 'Custom_Object_Detection_Gen3_ATSS',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            start_time: '2025-01-10T10:00:00.000000+00:00',
            end_time: '2025-01-10T12:30:00.000000+00:00',
            dataset_revision_id: 'dataset-1',
        },
    }),
    getMockedModel({
        id: 'model-rev-2',
        name: 'SSD Revision 1',
        architecture: 'Object_Detection_SSD',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            start_time: '2025-01-11T10:00:00.000000+00:00',
            end_time: '2025-01-11T12:30:00.000000+00:00',
            dataset_revision_id: 'dataset-1',
        },
    }),
];

const mockedDatasetRevisions = [
    getMockedDatasetRevision({ id: 'dataset-1', name: 'Dataset Revision 1' }),
    getMockedDatasetRevision({ id: 'dataset-2', name: 'Dataset Revision 2' }),
];

const runningTrainingJob = getMockedJob({
    job_id: 'job-train-1',
    job_type: 'train',
    status: 'RUNNING',
    progress: 25,
    message: 'Training in progress...',
    metadata: {
        project: { id: 'id-1' },
        model: {
            id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
            name: 'ATSS (ef3983f1)',
            architecture: 'Custom_Object_Detection_Gen3_ATSS',
            parent_revision_id: 'model-rev-1',
            dataset_revision_id: 'dataset-2',
        },
        device: {
            type: 'cpu',
            name: 'CPU',
        },
    },
    started_at: '2026-01-19T08:15:00.000000+00:00',
    finished_at: null,
});

const setupNetworkMocks = (network: NetworkFixture) => {
    let hasStartedTraining = false;

    const state: {
        submittedJobBody: TrainingRequestPayload | null;
        submittedTrainingConfiguration: TrainingConfigurationRequestPayload | null;
    } = {
        submittedJobBody: null,
        submittedTrainingConfiguration: null,
    };

    network.use(
        http.get('/api/projects/{project_id}/models', () => {
            return HttpResponse.json(mockedModelRevisions);
        }),
        http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
            if (params.model_id === 'model-training-1') {
                return HttpResponse.json(
                    getMockedModel({
                        id: 'model-training-1',
                        name: 'ATSS Training Run',
                        architecture: 'Custom_Object_Detection_Gen3_ATSS',
                        training_info: {
                            status: 'in_progress',
                            label_schema_revision: { labels: [] },
                            start_time: '2026-01-19T08:15:00.000000+00:00',
                            end_time: null,
                            dataset_revision_id: 'dataset-2',
                        },
                    })
                );
            }

            const model = mockedModelRevisions.find(({ id }) => id === params.model_id);
            return model ? HttpResponse.json(model) : new HttpResponse(null, { status: 404 });
        }),
        http.get('/api/projects/{project_id}/dataset_revisions', () => {
            return HttpResponse.json(mockedDatasetRevisions);
        }),
        http.get('/api/model_architectures', () => {
            return HttpResponse.json({
                model_architectures: mockedModelArchitectures,
                top_picks: {
                    balance: mockedModelArchitectures[0].id,
                    speed: mockedModelArchitectures[1].id,
                    accuracy: mockedModelArchitectures[1].id,
                },
            });
        }),
        http.get('/api/projects/{project_id}/dataset/items', () => {
            return HttpResponse.json({
                items: [
                    { id: '1', subset: 'training', user_reviewed: false },
                    { id: '2', subset: 'training', user_reviewed: false },
                    { id: '3', subset: 'validation', user_reviewed: false },
                    { id: '4', subset: 'testing', user_reviewed: false },
                ],
                pagination: { total: 4, count: 4, limit: 10, offset: 0 },
            });
        }),
        http.get('/api/jobs', () => {
            return HttpResponse.json(hasStartedTraining ? [runningTrainingJob] : []);
        }),
        http.post('/api/jobs', async ({ request }) => {
            state.submittedJobBody = (await request.json()) as TrainingRequestPayload;
            hasStartedTraining = true;
            return HttpResponse.json(runningTrainingJob, { status: 201 });
        }),
        http.get('/api/projects/{project_id}/training_configuration', () => {
            return HttpResponse.json(MOCKED_TRAINING_CONFIGURATION);
        }),
        http.get('/api/projects/{project_id}/models/{model_id}/training_configuration', () => {
            return HttpResponse.json(MOCKED_MODEL_TRAINING_CONFIGURATION);
        }),
        http.patch('/api/projects/{project_id}/training_configuration', async ({ request }) => {
            state.submittedTrainingConfiguration = await request.json();

            return HttpResponse.json({}, { status: 200 });
        })
    );

    return state;
};

test.describe('Model training flow', () => {
    test('Basic training flow', async ({ modelsPage, network, page }) => {
        const state = setupNetworkMocks(network);

        await modelsPage.goto();

        await modelsPage.openTrainModelDialog();
        await modelsPage.selectModelArchitecture('Custom_Object_Detection_Gen3_ATSS');
        await modelsPage.selectPickerOption('Select dataset', 'Dataset Revision 2');
        await modelsPage.selectPickerOption('Select model revision', 'ATSS Revision 1');

        await modelsPage.startTraining();

        await expect(page.getByRole('heading', { name: 'Currently running' })).toBeVisible();
        await expect(page.getByText('ATSS Training Run')).toBeVisible();

        expect(state.submittedJobBody).toMatchObject({
            job_type: 'train',
            project_id: 'id-1',
            parameters: {
                device: 'cpu',
                model_architecture_id: 'Custom_Object_Detection_Gen3_ATSS',
                dataset_revision_id: 'dataset-2',
                parent_model_revision_id: 'model-rev-1',
            },
        });
        expect(state.submittedTrainingConfiguration).toBeNull();
    });

    test('Advanced model training flow', async ({ network, modelsPage }) => {
        const state = setupNetworkMocks(network);

        await modelsPage.goto();
        await modelsPage.openTrainModelDialog();

        await test.step('Select model architecture', async () => {
            await modelsPage.selectModelArchitecture('Custom_Object_Detection_Gen3_ATSS');
        });

        await test.step('Select dataset revision', async () => {
            await modelsPage.selectPickerOption('Select dataset', 'Dataset Revision 2');
        });

        await test.step('Select model revision', async () => {
            await modelsPage.selectPickerOption('Select model revision', 'ATSS Revision 1');
        });

        await test.step('Select advanced settings', async () => {
            await modelsPage.openAdvancedSettings();
        });

        await test.step('Update any parameter', async () => {
            await modelsPage.openTrainingParameters();
            await modelsPage.updateInputSizeParameters(640, 512);
        });

        await test.step('Start the training', async () => {
            await modelsPage.startTraining();
        });

        expect(state.submittedJobBody).toMatchObject({
            job_type: 'train',
            project_id: 'id-1',
            parameters: {
                device: 'cpu',
                model_architecture_id: 'Custom_Object_Detection_Gen3_ATSS',
                dataset_revision_id: 'dataset-2',
                parent_model_revision_id: 'model-rev-1',
            },
        });

        const trainingParameters =
            findGroupByKey(MOCKED_MODEL_TRAINING_CONFIGURATION.parameters, 'training')?.parameters ?? [];

        const inputSizeWidthParameter = trainingParameters.find(
            isInputSizeWidthParameter
        ) as NumberEnumConfigurableParameter;
        const inputSizeHeightParameter = trainingParameters.find(
            isInputSizeHeightParameter
        ) as NumberEnumConfigurableParameter;

        const updatedTrainingConfigurationParameters = deepReplaceParameters(
            MOCKED_MODEL_TRAINING_CONFIGURATION.parameters,
            [
                { ...inputSizeWidthParameter, value: 640 },
                { ...inputSizeHeightParameter, value: 512 },
            ],
            ['training']
        );

        expect(state.submittedTrainingConfiguration).toMatchObject(
            getTrainingConfigurationUpdatePayload({ parameters: updatedTrainingConfigurationParameters })
        );
    });

    test('Advanced model training flow with training from scratch', async ({ network, modelsPage }) => {
        const state = setupNetworkMocks(network);

        await modelsPage.goto();
        await modelsPage.openTrainModelDialog();

        await test.step('Select model architecture', async () => {
            await modelsPage.selectModelArchitecture('Custom_Object_Detection_Gen3_ATSS');
        });

        await test.step('Select dataset revision', async () => {
            await modelsPage.selectPickerOption('Select dataset', 'Dataset Revision 2');
        });

        await test.step('Select model revision', async () => {
            await modelsPage.selectPickerOption('Select model revision', 'Train from scratch');
        });

        await test.step('Select advanced settings', async () => {
            await modelsPage.openAdvancedSettings();
        });

        await test.step('Update any parameter', async () => {
            await modelsPage.openTrainingParameters();
            await modelsPage.updateInputSizeParameters(640, 512);
        });

        await test.step('Start the training', async () => {
            await modelsPage.startTraining();
        });

        expect(state.submittedJobBody).toMatchObject({
            job_type: 'train',
            project_id: 'id-1',
            parameters: {
                device: 'cpu',
                model_architecture_id: 'Custom_Object_Detection_Gen3_ATSS',
                dataset_revision_id: 'dataset-2',
                parent_model_revision_id: null,
            },
        });

        const trainingParameters =
            findGroupByKey(MOCKED_MODEL_TRAINING_CONFIGURATION.parameters, 'training')?.parameters ?? [];

        const inputSizeWidthParameter = trainingParameters.find(
            isInputSizeWidthParameter
        ) as NumberEnumConfigurableParameter;
        const inputSizeHeightParameter = trainingParameters.find(
            isInputSizeHeightParameter
        ) as NumberEnumConfigurableParameter;

        const updatedTrainingConfigurationParameters = deepReplaceParameters(
            MOCKED_TRAINING_CONFIGURATION.parameters,
            [
                { ...inputSizeWidthParameter, value: 640 },
                { ...inputSizeHeightParameter, value: 512 },
            ],
            ['training']
        );

        expect(state.submittedTrainingConfiguration).toMatchObject(
            getTrainingConfigurationUpdatePayload({ parameters: updatedTrainingConfigurationParameters })
        );
    });
});
