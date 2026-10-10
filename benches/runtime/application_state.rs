use super::*;

const ITERATIONS: usize = if cfg!(debug_assertions) { 10 } else { 500 };

fn pure(root: Flow<u32>) -> Flow<u32> {
    root.map(|input| input + 1)
}

/// Separates explicit state boundary costs from instruction execution and payload sharing.
pub(super) fn report() -> Result<(), GraphError> {
    let flow = compile(pure)?;
    let mut runtime = flow.start_with_state(1, 0u32, uuid::Uuid::nil())?;
    report_allocations("application_state/start", || {
        flow.start_with_state(1, 0u32, uuid::Uuid::nil())
    });
    report_allocations("application_state/get", || runtime.get_state::<u32>());
    report_allocations("application_state/set", || runtime.set_state(1u32));
    report_sync("application_state/start", ITERATIONS, || {
        flow.start_with_state(1, 0u32, uuid::Uuid::nil())
    });
    report_sync("application_state/get", ITERATIONS, || {
        runtime.get_state::<u32>()
    });
    report_sync("application_state/set", ITERATIONS, || {
        runtime.set_state(1u32)
    });
    let snapshot = runtime.snapshot()?;
    report_allocations("application_state/snapshot", || runtime.snapshot());
    report_sync("application_state/snapshot", ITERATIONS, || {
        runtime.snapshot()
    });
    report_sync("application_state/restore", ITERATIONS, || {
        flow.restore(snapshot.clone())
    });
    report_steps(&flow)?;
    report_payload(&flow)
}

/// Measures an identical VM instruction with initialization and allocation outside timing.
fn report_steps(flow: &CompiledFlow<u32, u32>) -> Result<(), GraphError> {
    for with_state in [false, true] {
        let label = if with_state {
            "application_state/step"
        } else {
            "application_state/no_state_step"
        };
        let mut samples = Vec::new();
        for _ in 0..SYNC_SAMPLES {
            let mut executions = (0..VM_ITERATIONS)
                .map(|_| {
                    if with_state {
                        flow.start_with_state(1, 0u32, uuid::Uuid::nil())
                    } else {
                        flow.start(1, uuid::Uuid::nil())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let start = Instant::now();
            for runtime in &mut executions {
                black_box(runtime.next()?);
            }
            samples.push(ns_per_iteration(start.elapsed(), VM_ITERATIONS));
        }
        print_median(label, VM_ITERATIONS, SYNC_SAMPLES, &mut samples);
    }
    Ok(())
}

/// Snapshot capture shares the nested application payload instead of traversing its contents.
fn report_payload(flow: &CompiledFlow<u32, u32>) -> Result<(), GraphError> {
    let runtime = flow.start_with_state(1, vec!["state".repeat(1000); 128], uuid::Uuid::nil())?;
    report_allocations("application_state/large_snapshot", || runtime.snapshot());
    report_sync("application_state/large_snapshot", ITERATIONS, || {
        runtime.snapshot()
    });
    Ok(())
}
