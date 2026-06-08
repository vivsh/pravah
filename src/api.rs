

// Fluent Node API — usage reference
//
// Implement Flow::define instead of Flow::build:
//
//   impl Flow for SummariseRequest {
//       type Output = Report;
//       fn define(root: Node<Self>) -> FlowBuilder {
//           root.agent()
//               .work(format_bullets)
//               .finalize()
//       }
//   }
//
// Split / merge:
//
//   fn define(root: Node<Self>) -> FlowBuilder {
//       let (tech, mkt, risk) = root
//           .agent()
//           .split(|p| (TechTrack { .. }, MktTrack { .. }, RiskTrack { .. }));
//       let tech = tech.agent();
//       let mkt  = mkt.agent();
//       let risk = risk.agent();
//       tech.merge((mkt, risk), |(t, m, r)| Brief { .. })
//           .finalize()
//   }
//
// Agent with tools:
//
//   fn define(root: Node<Self>) -> FlowBuilder {
//       root.agent_with(|toolbox| {
//               toolbox
//                   .tool::<SearchTool>()
//                   .tool_flow::<VerifyClaim>()
//           })
//           .work(finish)
//           .finalize()
//   }
//
// Sidecar (orphan branch carried through):
//
//   fn define(root: Node<Self>) -> FlowBuilder {
//       let (main, sidecar) = root.split(|x| (x.clone(), x));
//       let main    = main.agent();
//       let sidecar = sidecar.work(|x| async { audit(x).await });
//       main.merge(sidecar, |(post, _log)| post)
//           .finalize()
//       // OR if sidecar result is not needed:
//       // main.hold(sidecar).finalize()
//   }
