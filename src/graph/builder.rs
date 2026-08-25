use serde_json::Value as JsonValue;

use super::error::GraphError;
use super::ids::{EdgeId, MarkId, NodeId, VarId};
use super::model::{
    Edge, Mark, Node, NodeKind, TypeSpec, UNTYPED_GRAPH_SCHEMA_VERSION, UntypedGraph, VarInit,
    VarKey, VarScope, Variable,
};
use super::validation::validate_graph_shape;
use super::value::Value;

#[derive(Debug)]
/// Imperative builder for deterministic untyped graphs.
///
/// Use this as the common backend for generated graphs, JSON loaders, and
/// typed fluent frontends. It accumulates wiring errors and reports them at
/// `build()`.
pub struct UntypedGraphBuilder {
    name: String,
    edges: Vec<Edge>,
    variables: Vec<Variable>,
    marks: Vec<Mark>,
    nodes: Vec<Node>,
    entry: Option<EdgeId>,
    exit: Option<EdgeId>,
    errors: Vec<String>,
}

impl UntypedGraphBuilder {
    /// Starts a new graph with a stable display/build name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            edges: Vec::new(),
            variables: Vec::new(),
            marks: Vec::new(),
            nodes: Vec::new(),
            entry: None,
            exit: None,
            errors: Vec::new(),
        }
    }

    /// Adds a labelled typed value edge and returns its dense id.
    pub fn edge(&mut self, label: impl Into<String>, type_spec: TypeSpec) -> EdgeId {
        let id = EdgeId(self.edges.len());
        self.edges.push(Edge {
            id,
            label: Some(label.into()),
            type_spec,
            producer: None,
            consumers: Vec::new(),
        });
        id
    }

    /// Adds an unlabeled typed value edge for generated internal wiring.
    pub fn anonymous_edge(&mut self, type_spec: TypeSpec) -> EdgeId {
        let id = EdgeId(self.edges.len());
        self.edges.push(Edge {
            id,
            label: None,
            type_spec,
            producer: None,
            consumers: Vec::new(),
        });
        id
    }

    /// Declares a graph variable with explicit key, scope, and init policy.
    pub fn variable(
        &mut self,
        key: VarKey,
        type_spec: TypeSpec,
        scope: VarScope,
        init: VarInit,
    ) -> VarId {
        let id = VarId(self.variables.len());
        self.variables.push(Variable {
            id,
            key,
            type_spec,
            scope,
            init,
        });
        id
    }

    /// Declares a graph variable initialized from a JSON value.
    pub fn variable_with_value(
        &mut self,
        key: VarKey,
        type_spec: TypeSpec,
        scope: VarScope,
        value: Value,
    ) -> VarId {
        self.variable(key, type_spec, scope, VarInit::Value(value))
    }

    /// Declares a typed re-entry mark on an existing edge.
    pub fn mark(&mut self, edge: EdgeId) -> MarkId {
        self.mark_with_label(None::<String>, edge)
    }

    /// Declares a labelled typed re-entry mark on an existing edge.
    pub fn mark_with_label(&mut self, label: Option<impl Into<String>>, edge: EdgeId) -> MarkId {
        let id = MarkId(self.marks.len());
        let type_spec = match self.get_edge(edge) {
            Some(edge) => edge.type_spec.clone(),
            None => {
                self.errors.push(format!(
                    "mark {:?} references missing target edge {:?}",
                    id, edge
                ));
                TypeSpec::new("<invalid>", JsonValue::Null)
            }
        };
        self.marks.push(Mark {
            id,
            label: label.map(Into::into),
            target: edge,
            type_spec,
        });
        id
    }

    /// Adds a goto node that writes its input value to a declared mark.
    pub fn goto(&mut self, name: impl Into<String>, input: EdgeId, mark: MarkId) -> NodeId {
        let name = name.into();
        match (self.get_edge(input), self.marks.get(mark.0)) {
            (Some(input_edge), Some(mark_data)) => {
                if mark_data.id != mark {
                    self.errors.push(format!(
                        "goto node '{name}' references mark {:?} stored at a mismatched slot",
                        mark
                    ));
                } else if input_edge.type_spec != mark_data.type_spec {
                    self.errors.push(format!(
                        "goto node '{name}' input type '{}' does not match mark target type '{}'",
                        input_edge.type_spec.name, mark_data.type_spec.name
                    ));
                }
            }
            (None, _) => self.errors.push(format!(
                "goto node '{name}' references missing input edge {input:?}"
            )),
            (_, None) => self.errors.push(format!(
                "goto node '{name}' references missing mark {mark:?}"
            )),
        }
        self.node(name, NodeKind::Goto { mark }, vec![input], Vec::new())
    }

    /// Sets the graph input edge.
    ///
    /// The edge must already exist; invalid ids are recorded and surfaced by
    /// `build()`.
    pub fn set_entry(&mut self, edge: EdgeId) -> &mut Self {
        if self.get_edge(edge).is_none() {
            self.errors
                .push(format!("entry edge {:?} does not exist", edge));
        }
        self.entry = Some(edge);
        self
    }

    /// Sets the graph output edge.
    ///
    /// The edge must already exist; invalid ids are recorded and surfaced by
    /// `build()`.
    pub fn set_exit(&mut self, edge: EdgeId) -> &mut Self {
        if self.get_edge(edge).is_none() {
            self.errors
                .push(format!("exit edge {:?} does not exist", edge));
        }
        self.exit = Some(edge);
        self
    }

    /// Adds an operation node and wires its input/output edge metadata.
    ///
    /// Duplicate output producers and missing edge ids are collected as build
    /// errors instead of panicking.
    pub fn node(
        &mut self,
        name: impl Into<String>,
        kind: NodeKind,
        inputs: Vec<EdgeId>,
        outputs: Vec<EdgeId>,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        let name = name.into();

        for input in &inputs {
            match self.edge_mut(*input) {
                Some(edge) => edge.consumers.push(id),
                None => self.errors.push(format!(
                    "node '{name}' references missing input edge {input:?}"
                )),
            }
        }

        for output in &outputs {
            match self.edge_mut(*output) {
                Some(edge) => {
                    if let Some(existing) = edge.producer {
                        self.errors.push(format!(
                            "edge {:?} already has producer {:?}; node '{name}' cannot also produce it",
                            output, existing
                        ));
                    } else {
                        edge.producer = Some(id);
                    }
                }
                None => self.errors.push(format!(
                    "node '{name}' references missing output edge {output:?}"
                )),
            }
        }

        self.nodes.push(Node {
            id,
            name,
            kind,
            inputs,
            outputs,
        });
        id
    }

    /// Finalizes and validates the graph.
    ///
    /// This is the single place where accumulated builder errors and graph
    /// shape errors become `GraphError`s.
    pub fn build(self) -> Result<UntypedGraph, GraphError> {
        if !self.errors.is_empty() {
            return Err(GraphError::Invalid(self.errors.join("; ")));
        }
        let entry = self
            .entry
            .ok_or_else(|| GraphError::Invalid("entry edge is not set".into()))?;
        let exit = self
            .exit
            .ok_or_else(|| GraphError::Invalid("exit edge is not set".into()))?;
        let graph = UntypedGraph {
            schema_version: UNTYPED_GRAPH_SCHEMA_VERSION,
            name: self.name,
            edges: self.edges,
            variables: self.variables,
            marks: self.marks,
            nodes: self.nodes,
            entry,
            exit,
        };
        validate_graph_shape(&graph)?;
        Ok(graph)
    }

    fn get_edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.0).filter(|edge| edge.id == id)
    }

    fn edge_mut(&mut self, id: EdgeId) -> Option<&mut Edge> {
        self.edges.get_mut(id.0).filter(|edge| edge.id == id)
    }
}
