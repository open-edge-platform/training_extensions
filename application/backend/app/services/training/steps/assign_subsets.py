# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy.orm import Session

from app.core.run import ExecutionContext
from app.services.training.base import PipelineContext, TrainingStep
from app.services.training.models import TrainingParams

from .subset_assignment import SplitRatios, SubsetAssigner, SubsetService

logger = logging.getLogger(__name__)


class AssignSubsetsStep(TrainingStep):
    def __init__(
        self,
        subset_service: SubsetService,
        assigner: SubsetAssigner,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ) -> None:
        self._subset_service = subset_service
        self._assigner = assigner
        self._db_session_factory = db_session_factory

    def execute(self, ctx: ExecutionContext, params: TrainingParams, _pipeline_ctx: PipelineContext) -> None:
        """Assigning subsets to all unassigned dataset items in the project dataset."""
        ctx.report_progress("Retrieving unassigned items")
        project_id = params.project_id
        if project_id is None:
            raise ValueError("Project ID must be provided for subset assignment")
        with self._db_session_factory() as db:
            unassigned_items = self._subset_service.get_unassigned_items_with_labels(project_id, db)

            if not unassigned_items:
                ctx.report_progress("No unassigned items found")
                return

            ctx.report_progress(f"Found {len(unassigned_items)} unassigned items")

            # Get current distribution
            current_distribution = self._subset_service.get_subset_distribution(project_id, db)
            logger.info("Current subset distribution: %s", current_distribution)

            # Compute adjusted ratios
            # TODO: Infer target ratios from training params
            target_ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
            adjusted_ratios = current_distribution.compute_adjusted_ratios(target_ratios, len(unassigned_items))

            ctx.report_progress("Computing optimal subset assignments")
            assignments = self._assigner.assign(unassigned_items, adjusted_ratios)

            # Persist assignments
            ctx.report_progress("Persisting subset assignments")
            self._subset_service.update_subset_assignments(project_id, assignments, db)

        ctx.report_progress(f"Successfully assigned {len(assignments)} items to subsets")

    def get_name(self) -> str:
        return "Assign Subsets"
