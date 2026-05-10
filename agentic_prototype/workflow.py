from __future__ import annotations

from typing import Dict, TypedDict

from langgraph.graph import END, StateGraph

from .agents import AgentOutput, CoordinatorAgent, RiskAgent, SentimentMacroAgent, TechnicalAgent
from .skills import build_default_registry


class WorkflowState(TypedDict, total=False):
    case: Dict
    technical: AgentOutput
    sentiment: AgentOutput
    risk: AgentOutput
    final: AgentOutput


def build_workflow():
    skills = build_default_registry()
    technical_agent = TechnicalAgent(name="Technical Agent", skills=skills)
    sentiment_agent = SentimentMacroAgent(name="Sentiment+Macro Agent", skills=skills)
    risk_agent = RiskAgent(name="Risk Agent", skills=skills)
    coordinator = CoordinatorAgent(name="Coordinator Agent", skills=skills)

    graph = StateGraph(WorkflowState)

    def technical_node(state: WorkflowState) -> WorkflowState:
        return {"technical": technical_agent.analyze(state["case"])}

    def sentiment_node(state: WorkflowState) -> WorkflowState:
        return {"sentiment": sentiment_agent.analyze(state["case"])}

    def risk_node(state: WorkflowState) -> WorkflowState:
        return {"risk": risk_agent.analyze(state["case"])}

    def coordinator_node(state: WorkflowState) -> WorkflowState:
        final = coordinator.decide(state["technical"], state["sentiment"], state["risk"])
        return {"final": final}

    graph.add_node("technical", technical_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("risk", risk_node)
    graph.add_node("coordinator", coordinator_node)

    graph.set_entry_point("technical")
    graph.add_edge("technical", "sentiment")
    graph.add_edge("sentiment", "risk")
    graph.add_edge("risk", "coordinator")
    graph.add_edge("coordinator", END)

    return graph.compile()


def run_case(case: Dict) -> Dict:
    app = build_workflow()
    result = app.invoke({"case": case})
    final = result["final"]
    return {
        "id": case["id"],
        "predicted_signal": final.signal,
        "predicted_score": round(final.score, 4),
        "predicted_confidence": round(final.confidence, 4),
        "reasoning": final.reasoning,
        "expected_signal": case.get("expected_signal"),
    }

