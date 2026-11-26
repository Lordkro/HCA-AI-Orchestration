//! The orchestrator's [`UsageMeter`]: one per agent run.
//!
//! - **Reservations are saved before each call.** The call's worst case
//!   ([`hca_llm::max_request_tokens`]) is reserved in the database, against
//!   the project's spent tokens plus every reservation still open. Parallel
//!   runs can't spend more than the budget between them.
//! - **Every reservation is settled**, with the call's real cost, or in full
//!   when the cost is unknown: the call timed out, the run was cut short
//!   mid-call, or the process died (then at the next start). No billed
//!   token goes uncounted.
//! - **A call that can never fit ends the project** as out of budget at
//!   once, instead of failing tasks one retry at a time. A call that only
//!   waits on other calls' reservations waits for them to settle.
//! - **Paused, finished or shutting down stops the run** at the next call.
//!
//! Dropping the meter (the run ended, or its future was dropped) settles
//! what it still holds and closes the run on the status board.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::PoisonError;
use std::time::Duration;

use chrono::Utc;
use hca_agents::{MeterError, UsageMeter};
use hca_core::{AgentRole, ProjectId, ProjectStatus, Reservation, TaskId};
use hca_llm::{BoxFuture, Usage};
use hca_store::StoreError;

use crate::Orchestrator;

/// What a stopped run says when its project is paused.
pub(crate) const STOP_PAUSED: &str = "the project was paused";
/// What a stopped run says when HCA is shutting down.
pub(crate) const STOP_SHUTDOWN: &str = "HCA is shutting down";
/// What a stopped run says when the budget can't cover its next call.
pub(crate) const STOP_BUDGET: &str = "the remaining token budget can't cover the next model call";

/// How often a call waiting on other calls' reservations asks again.
const BUSY_RETRY: Duration = Duration::from_millis(250);

pub(crate) struct ProjectMeter<'a> {
    orchestrator: &'a Orchestrator,
    project: ProjectId,
    task: Option<TaskId>,
    role: AgentRole,
    run_id: u64,
    /// This run's open reservation, settled in full on drop.
    held: AtomicU64,
    tokens: AtomicU64,
    calls: AtomicU64,
    exhausted: AtomicBool,
}

impl<'a> ProjectMeter<'a> {
    pub(crate) fn new(
        orchestrator: &'a Orchestrator,
        project: ProjectId,
        task: Option<TaskId>,
        role: AgentRole,
        run_id: u64,
    ) -> Self {
        Self {
            orchestrator,
            project,
            task,
            role,
            run_id,
            held: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            exhausted: AtomicBool::new(false),
        }
    }

    async fn reserve_now(&self, estimate: u64) -> Result<(), MeterError> {
        let o = self.orchestrator;
        loop {
            if o.stopping.load(Ordering::SeqCst) {
                return Err(MeterError::Stopped(STOP_SHUTDOWN.into()));
            }
            let reservation =
                o.store.reserve_usage(self.project, estimate, &o.config.limits, Utc::now()).await;
            match reservation {
                Ok(Reservation::Granted) => {
                    self.held.fetch_add(estimate, Ordering::SeqCst);
                    return Ok(());
                }
                Ok(Reservation::Busy) => tokio::time::sleep(BUSY_RETRY).await,
                Ok(Reservation::Refused(error)) => {
                    tracing::info!(project = %self.project, %error, "project out of budget");
                    if !self.exhausted.swap(true, Ordering::SeqCst) {
                        o.after_project_change(self.project, "Token budget spent. Project stopped.").await;
                    }
                    return Err(MeterError::Stopped(STOP_BUDGET.into()));
                }
                Err(StoreError::ProjectNotActive(ProjectStatus::Paused)) => {
                    return Err(MeterError::Stopped(STOP_PAUSED.into()))
                }
                Err(StoreError::ProjectNotActive(other)) => {
                    return Err(MeterError::Stopped(format!(
                        "the project is {}",
                        format!("{other:?}").to_lowercase()
                    )))
                }
                Err(error) => return Err(MeterError::Failed(error.to_string())),
            }
        }
    }

    async fn settle_now(&self, estimate: u64, usage: Option<Usage>) -> Result<(), MeterError> {
        let o = self.orchestrator;
        let charged = usage.map_or(estimate, |usage| usage.total());
        let within = o
            .store
            .settle_usage(self.project, self.task, estimate, charged, &o.config.limits, Utc::now())
            .await
            .map_err(|e| MeterError::Failed(e.to_string()))?;
        self.held.fetch_sub(estimate.min(self.held.load(Ordering::SeqCst)), Ordering::SeqCst);
        self.tokens.fetch_add(charged, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !within && !self.exhausted.swap(true, Ordering::SeqCst) {
            o.after_project_change(self.project, "Token budget spent. Project stopped.").await;
        }
        Ok(())
    }
}

impl UsageMeter for ProjectMeter<'_> {
    fn reserve(&self, estimate: u64) -> BoxFuture<'_, Result<(), MeterError>> {
        Box::pin(self.reserve_now(estimate))
    }

    fn settle(&self, estimate: u64, usage: Option<Usage>) -> BoxFuture<'_, Result<(), MeterError>> {
        Box::pin(self.settle_now(estimate, usage))
    }
}

impl Drop for ProjectMeter<'_> {
    fn drop(&mut self) {
        // Still holding a reservation means the run was cut off mid-call (a
        // run timeout, or shutdown), so the call may have been billed:
        // charge it in full. If this can't run, the next start does it.
        let held = *self.held.get_mut();
        if held > 0 {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                let (store, limits, project, task) = (
                    self.orchestrator.store.clone(),
                    self.orchestrator.config.limits,
                    self.project,
                    self.task,
                );
                runtime.spawn(async move {
                    if let Err(error) = store.settle_usage(project, task, held, held, &limits, Utc::now()).await {
                        tracing::warn!(%error, "could not settle a cut-off model call; it is charged at the next start");
                    }
                });
            }
        }
        let (calls, tokens) = (*self.calls.get_mut(), *self.tokens.get_mut());
        self.orchestrator.board.lock().unwrap_or_else(PoisonError::into_inner).finish(
            self.role,
            self.run_id,
            calls,
            tokens,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use hca_core::{Limits, Project, ProjectEvent, ProjectFailure};
    use hca_llm::Llm;
    use hca_store::Store;

    use super::*;
    use crate::OrchestratorConfig;

    async fn orchestrator(budget: u64) -> (Orchestrator, Project, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("hca.db")).await.unwrap();
        let project = Project::new("p", "idea", Utc::now());
        store.create_project(&project).await.unwrap();
        let mut config = OrchestratorConfig::new(dir.path().join("workspaces"));
        config.limits = Limits { project_token_budget: budget, ..Limits::default() };
        (Orchestrator::new(store, Llm::new(HashMap::new()), None, config), project, dir)
    }

    async fn spent(o: &Orchestrator, project: ProjectId) -> (u64, u64) {
        let p = o.store.project(project).await.unwrap();
        (p.tokens_used(), p.tokens_reserved())
    }

    #[tokio::test]
    async fn parallel_runs_cannot_overspend_together() {
        let (o, project, _dir) = orchestrator(1_000).await;
        let first = ProjectMeter::new(&o, project.id, None, AgentRole::Coder, 1);
        let second = ProjectMeter::new(&o, project.id, None, AgentRole::Coder, 2);

        first.reserve(700).await.unwrap();
        assert_eq!(spent(&o, project.id).await, (0, 700), "saved before the call is made");
        // The second call fits the budget, just not next to the first one:
        // it waits until the first settles.
        let waiting = second.reserve(700);
        let settle = async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            first.settle(700, Some(Usage { input_tokens: 200, output_tokens: 100 })).await.unwrap();
        };
        let (granted, ()) =
            tokio::time::timeout(Duration::from_secs(10), async { tokio::join!(waiting, settle) })
                .await
                .expect("the waiting call went ahead");
        granted.unwrap();
        assert_eq!(spent(&o, project.id).await, (300, 700));

        second.settle(700, None).await.unwrap();
        assert_eq!(spent(&o, project.id).await, (1_000, 0), "an unknown cost is charged in full");
        assert_eq!(
            o.store.project(project.id).await.unwrap().failure(),
            Some(&ProjectFailure::BudgetExhausted)
        );
    }

    #[tokio::test]
    async fn a_run_cut_off_mid_call_is_charged_in_full() {
        let (o, project, _dir) = orchestrator(10_000).await;
        let meter = ProjectMeter::new(&o, project.id, None, AgentRole::Coder, 1);
        meter.reserve(700).await.unwrap();
        drop(meter);
        for _ in 0..100 {
            if spent(&o, project.id).await == (700, 0) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("not charged: {:?}", spent(&o, project.id).await);
    }

    #[tokio::test]
    async fn a_call_that_can_never_fit_stops_the_project() {
        let (o, project, _dir) = orchestrator(1_000).await;
        let meter = ProjectMeter::new(&o, project.id, None, AgentRole::Coder, 1);
        let refused = meter.reserve(1_001).await.unwrap_err();
        assert!(matches!(&refused, MeterError::Stopped(r) if r == STOP_BUDGET), "{refused:?}");
        let p = o.store.project(project.id).await.unwrap();
        assert_eq!(
            (p.status(), p.failure()),
            (ProjectStatus::Failed, Some(&ProjectFailure::BudgetExhausted))
        );
        assert_eq!(p.tokens_used(), 0, "nothing was spent");
    }

    #[tokio::test]
    async fn paused_projects_and_shutdown_stop_runs() {
        let (o, project, _dir) = orchestrator(1_000).await;
        let meter = ProjectMeter::new(&o, project.id, None, AgentRole::Coder, 1);
        o.store.apply_project_event(project.id, ProjectEvent::Pause, Utc::now()).await.unwrap();
        assert!(matches!(meter.reserve(1).await, Err(MeterError::Stopped(r)) if r == STOP_PAUSED));

        o.store.apply_project_event(project.id, ProjectEvent::Resume, Utc::now()).await.unwrap();
        meter.reserve(1).await.unwrap();
        meter.settle(1, Some(Usage::default())).await.unwrap();
        o.begin_shutdown();
        assert!(matches!(meter.reserve(1).await, Err(MeterError::Stopped(r)) if r == STOP_SHUTDOWN));
    }
}
