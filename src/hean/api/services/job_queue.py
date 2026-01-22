"""Job queue service for async tasks (backtest, evaluate)."""

import asyncio
import uuid
from datetime import datetime
from typing import Any

from hean.api.schemas import JobStatus
from hean.logging import get_logger

logger = get_logger(__name__)


class Job:
    """Job representation."""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        params: dict[str, Any],
    ) -> None:
        """Initialize job."""
        self.job_id = job_id
        self.job_type = job_type
        self.params = params
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.progress = 0.0
        self._task: asyncio.Task[None] | None = None


class JobQueueService:
    """Service for managing async jobs."""

    def __init__(self) -> None:
        """Initialize job queue service."""
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def submit_job(
        self,
        job_type: str,
        params: dict[str, Any],
        task_fn: Any,
    ) -> str:
        """Submit a new job.

        Args:
            job_type: Type of job (backtest, evaluate, etc.)
            params: Job parameters
            task_fn: Async function to execute

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = Job(job_id, job_type, params)

        async with self._lock:
            self._jobs[job_id] = job

        # Start task
        job._task = asyncio.create_task(self._run_job(job, task_fn))

        logger.info(f"Submitted job {job_id} of type {job_type}")
        return job_id

    async def _run_job(self, job: Job, task_fn: Any) -> None:
        """Run job task."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()

        try:
            result = await task_fn(job)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            logger.info(f"Job {job.job_id} completed successfully")
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)

    async def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_jobs(self, limit: int = 100) -> list[Job]:
        """List recent jobs."""
        async with self._lock:
            jobs = list(self._jobs.values())
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False

            if job._task:
                job._task.cancel()
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            logger.info(f"Job {job_id} cancelled")
            return True

    def to_dict(self, job: Job) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result": job.result,
            "error": job.error,
            "progress": job.progress,
        }


# Global instance
job_queue_service = JobQueueService()

