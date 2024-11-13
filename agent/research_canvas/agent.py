"""
This is the main entry point for the AI.
It defines the workflow graph and the entry point for the agent.
"""
# pylint: disable=line-too-long, unused-import
import json
from typing import cast

from langchain_core.messages import AIMessage, ToolMessage,HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from research_canvas.state import AgentState
from research_canvas.download import download_node
from research_canvas.chat import chat_node
from research_canvas.search import search_node
from research_canvas.delete import delete_node, perform_delete_node
from research_canvas.arxiv import arxiv_agent_node

# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("download", download_node)
workflow.add_node("chat_node", chat_node)
workflow.add_node("search_node", search_node)
workflow.add_node("delete_node", delete_node)
workflow.add_node("perform_delete_node", perform_delete_node)
workflow.add_node("arxiv_node", arxiv_agent_node)  # Add this new node

def route(state):
    """Route after the chat node."""
    messages = state.get("messages", [])
    
    # Check for AI message with tool calls
    if messages and isinstance(messages[-1], AIMessage):
        ai_message = cast(AIMessage, messages[-1])

        if ai_message.tool_calls:
            tool_name = ai_message.tool_calls[0]["name"]
            if tool_name == "Search":
                return "search_node"
            elif tool_name == "DeleteResources":
                return "delete_node"
            elif tool_name == "search_arxiv":  # Add this condition
                return "arxiv_node"

    # Check for direct arxiv search commands in human messages
    if messages and isinstance(messages[-1], HumanMessage):
        if "search_papers" in messages[-1].content.lower():
            return "arxiv_node"
            
    # Check for tool messages
    if messages and isinstance(messages[-1], ToolMessage):
        return "chat_node"

    return END


memory = MemorySaver()
workflow.set_entry_point("download")
workflow.add_edge("download", "chat_node")
workflow.add_conditional_edges(
    "chat_node",
    route,
    ["search_node", "chat_node", "delete_node", "arxiv_node", END]
)
workflow.add_edge("delete_node", "perform_delete_node")
workflow.add_edge("perform_delete_node", "chat_node")
workflow.add_edge("search_node", "download")
workflow.add_edge("arxiv_node", "chat_node")
graph = workflow.compile(checkpointer=memory, interrupt_after=["delete_node"])
