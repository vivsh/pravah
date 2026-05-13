use std::{collections::{HashMap, VecDeque}, sync::Arc};


use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub struct AgentNode{

}

pub struct FlowNode{

}

pub struct ToolNode{

}

pub struct EitherNode{

}

pub struct GotoNode{

}

pub struct ForkNode{

}

pub struct JoinNode{

}

pub struct ReturnNode{

}

pub struct SuspendNode{
    output_index: usize,
}

pub struct CallNode{
    callable_index: usize,
    return_index: usize,
}


#[derive(Error, Debug)]
pub enum StepError{

    #[error("Invalid callable index")]
    InvalidCallable,

    #[error("Callable does not have an operation at the given index")]
    CallableNotFound(usize),

    #[error("Operation not found at the given index")]
    OperationNotFound(usize),

    #[error("Maximum frame depth exceeded")]
    FrameDepthExceeded,

    #[error("Invalid suspension state")]
    SuspendError(String),

    #[error("Error during resumption: {0}")]
    ResumeError(String),

    #[error("Error during return: {0}")]
    ReturnError(String),

    #[error("Execution completed with final output")]
    ExitError,

}

pub enum Operation{
    Call(CallNode), // index of the callable to call
    Tool(ToolNode),
    Either(EitherNode),
    Goto(GotoNode),
    Fork(ForkNode),
    Join(JoinNode),
    Return(ReturnNode),
    Suspend(SuspendNode),
}

impl Operation{
    
    pub fn input_index(&self)->usize{
        0
    }

    pub fn output_index(&self)->Option<usize>{
        None
    }

}

pub struct Instruction{
    inputs: Vec<usize>, // absolute locals indices for the inputs
    outputs: Vec<usize>, // absolute locals indices for the outputs
    operation: Operation,
    index: usize, // index of the instruction in the callable's instruction list
    next: usize, // index of the next instruction to execute after this one
}

pub enum Callable{
    Agent(AgentNode),
    Flow(FlowNode),
}

impl Callable{

    pub fn index(&self)->usize{
        0
    }

    pub fn frame_size(&self)->usize{
        match self{
            Callable::Agent(agent) => 0, // implement logic to get frame size for agent
            Callable::Flow(flow) => 0, // implement logic to get frame size for flow
        }
    }

    pub fn instruction(&self, index: usize)->Option<&Instruction>{
        match self{
            Callable::Agent(agent) => None, // implement logic to get operation for agent
            Callable::Flow(flow) => None, // implement logic to get operation for flow
        }        
    }

    pub async fn call(&self, frame: &mut Frame, locals: &mut Vec<Value>)->Result<&Instruction, StepError>{
        match self{
            Callable::Agent(agent) => self.call_agent(frame, agent, locals).await,
            Callable::Flow(flow) => self.call_flow(frame, flow, locals).await,
        }
    }

    async fn call_agent(&self, frame: &mut Frame, agent: &AgentNode, locals: &mut Vec<Value>)->Result<&Instruction, StepError>{
        unimplemented!()
    }

    async fn call_flow(&self, frame: &mut Frame, flow: &FlowNode, locals: &mut Vec<Value>)->Result<&Instruction, StepError>{
        unimplemented!()
    }

}


#[derive(Debug, Serialize, Deserialize)]
pub struct Frame{
    callable: usize,
    instruction: usize,
    return_to: usize, // absolute locals index to return the value to
    start: usize,
    size: usize,
    depth: usize,
}

impl Frame{

    pub fn new(callable: &Callable, locals_offset: usize, depth: usize, return_to: usize)->Self{
        Self{
            instruction: 0,
            start: locals_offset,
            size: callable.frame_size(),
            depth,
            return_to,
            callable: callable.index()
        }
    }

}


#[derive(Debug, Serialize, Deserialize)]
pub struct State{
    suspension: Option<usize>,
    locals: Vec<Value>,
    frames: Vec<Frame>,
}

impl State{

    pub fn new()->Self{
        Self{
            suspension: None,
            locals: Vec::new(),
            frames: Vec::new(),
        }
    }

}

pub struct Script{
    map: HashMap<String, usize>,
    callables: Vec<Callable>,
    entry: usize,
}

impl Script{

    pub fn new()->Self{
        Self{
            map: HashMap::new(),
            callables: Vec::new(),
            entry: 0,
        }
    }

    fn add_callable(&mut self, name: &str, callable: Callable)->usize{
        let index = self.callables.len();
        self.callables.push(callable);
        self.map.insert(name.to_string(), index);
        index
    }

    fn intern_str(&mut self, s: &str)->usize{
        if let Some(&index) = self.map.get(s){
            index
        }else{
            let index = self.map.len();
            self.map.insert(s.to_string(), index);
            index
        }
    }

}

pub struct VM{
    state: State,
    script: Script,
}



pub enum Step{
    Continue,
    Return(Value),
    Suspend(Value),
}

impl VM {  

    pub fn new(script: Script)->Self{
        Self{
            state: State::new(),
            script
        }
    }  

    pub fn resume(&mut self, value: Value)->Result<(), StepError>{
        if let Some(suspension) = self.state.suspension.take(){
            self.state.locals[suspension] = value; // store the resumed value in locals
            Ok(())
        }else{
            Err(StepError::ResumeError("No suspension to resume".to_string()))
        }
    }

    fn handle_return(&mut self, input_index: usize)->Result<Step, StepError>{
        if let Some(frame) = self.state.frames.pop(){
            let mut return_value = Value::Null;
            std::mem::swap(&mut return_value, &mut self.state.locals[input_index]);   
            if self.state.frames.is_empty(){
                // if there are no more frames, return the final output
                Ok(Step::Return(return_value))
            }else{
                std::mem::swap(&mut return_value, &mut self.state.locals[frame.return_to]);
                Ok(Step::Continue)
            }         
        }else{
            Err(StepError::ReturnError("No frame to return from".to_string()))
        }
    }

    fn handle_suspend(&mut self, output_index: usize, input_index: usize)->Result<Step, StepError>{
        let mut suspend_value = Value::Null;
        std::mem::swap(&mut suspend_value, &mut self.state.locals[input_index]);
        self.state.suspension = Some(output_index);
        Ok(Step::Suspend(suspend_value))
    }

    fn handle_call(&mut self, callable_index: usize, return_index: usize)->Result<Step, StepError>{
        if let Some(callable) = self.script.callables.get(callable_index){
            let locals_offset = self.state.locals.len();
            let new_frame = Frame::new(callable, locals_offset, self.state.frames.len(), return_index);
            self.state.frames.push(new_frame);
            Ok(Step::Continue)
        }else{
            Err(StepError::CallableNotFound(callable_index))
        }
    }
    
    pub async fn step(&mut self)->Result<Step, StepError>{        
        if let Some(_) = self.state.suspension{
            return Err(StepError::SuspendError("Cannot step while suspended".to_string()))
        }
        if let Some(frame) = self.state.frames.last_mut(){
            if let Some(callable) = &self.script.callables.get(frame.callable) {
                let ins = callable.call(frame, &mut self.state.locals).await?;    
                let input_index = ins.operation.input_index() + frame.start;            
                match ins.operation{
                    Operation::Call(ref cx) => {
                        return self.handle_call(cx.callable_index, cx.return_index);
                    },
                    Operation::Return(ReturnNode{}) => {
                        return self.handle_return(input_index);
                    },
                    Operation::Suspend(SuspendNode { output_index }) => {
                        let output_index = output_index + frame.start;
                        return self.handle_suspend(output_index, input_index);
                    },
                    _ => {}
                }

                return Ok(Step::Continue);
            } else {
                return Err(StepError::InvalidCallable);
                
            }
        } else {
            // no more frames to execute, return Step::Return with the final output
            return Err(StepError::ExitError)
        }
    }

}