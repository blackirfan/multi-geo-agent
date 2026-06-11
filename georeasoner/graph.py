"""Assemble the GeoReasoner LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from georeasoner.agents.gis_analyst import gis_analyst_node
from georeasoner.agents.hydrology import hydrology_node
from georeasoner.agents.planner import planner_node
from georeasoner.agents.reasoner import reasoner_node
from georeasoner.agents.remote_sensing import remote_sensing_node
from georeasoner.state import GeoReasonerState


def assemble_graph():
    """
    Build and compile the sequential GeoReasoner graph.

    Topology:
        START → planner → gis_analyst → remote_sensing → hydrology → reasoner → END
    """
    builder = StateGraph(GeoReasonerState)

    builder.add_node("planner", planner_node)
    builder.add_node("gis_analyst", gis_analyst_node)
    builder.add_node("remote_sensing", remote_sensing_node)
    builder.add_node("hydrology", hydrology_node)
    builder.add_node("reasoner", reasoner_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "gis_analyst")
    builder.add_edge("gis_analyst", "remote_sensing")
    builder.add_edge("remote_sensing", "hydrology")
    builder.add_edge("hydrology", "reasoner")
    builder.add_edge("reasoner", END)

    return builder.compile()
