

// Fluent Node API — usage reference
//
// Implement Flow::build with the fluent Node API:
//
//   impl Flow for SummariseRequest {
//       type Output = Report;
//       fn build(root: Node<Self>) -> Node<Self::Output> {
//           root.agent()
//               .work(format_bullets)
//       }
//   }
//
// Split / merge:
//
//   fn build(root: Node<Self>) -> Node<Self::Output> {
//       let (tech, mkt, risk) = root
//           .agent()
//           .split(|p| (TechTrack { .. }, MktTrack { .. }, RiskTrack { .. }));
//       let tech = tech.agent();
//       let mkt  = mkt.agent();
//       let risk = risk.agent();
//       tech.merge((mkt, risk), |(t, m, r)| Brief { .. })
//   }
//
// Agent with tools:
//
//   fn build(root: Node<Self>) -> Node<Self::Output> {
//       root.agent_with(|toolbox| {
//               toolbox
//                   .tool::<SearchTool>()
//                   .tool_flow::<VerifyClaim>()
//           })
//           .work(finish)
//   }
//
// Sidecar (orphan branch carried through):
//
//   fn build(root: Node<Self>) -> Node<Self::Output> {
//       let (main, sidecar) = root.split(|x| (x.clone(), x));
//       let main    = main.agent();
//       let sidecar = sidecar.work(|x| async { audit(x).await });
//       main.merge(sidecar, |(post, _log)| post)
//       // OR if sidecar result is not needed:
//       // main.hold(sidecar)
//   }
