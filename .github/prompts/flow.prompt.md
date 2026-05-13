## Flow Engine

- Works like an interpreter with stack of call-frame (each agent invocation is a call) and a global state.
- It can suspend and resume with a single global suspension variable.
- Currently only tools are allowed to suspend.
- On suspend, the suspension field should hold the exit_name or the node-id of Tool::Output. The value of ToolOutput::Suspend variant should be the tool itself.
- When resume is called, the resumption object's type should match this node.
- Upon resume, agent should be invoked and tool-calls completed as if the output came directly from the tool.
