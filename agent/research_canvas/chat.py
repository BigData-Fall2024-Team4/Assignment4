from typing import List, cast, Optional, Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain.tools import BaseTool, tool
from copilotkit.langchain import copilotkit_customize_config
from research_canvas.state import AgentState, Resource, Log
from research_canvas.model import get_model
from research_canvas.download import get_resource
import requests
import re
from pydantic import BaseModel, Field

class ArxivInput(BaseModel):
    """Input for ArXiv fetch."""
    arxiv_id: str = Field(description="The ArXiv ID of the paper to fetch")

class FetchArxivTool(BaseTool):
    name: str = "fetch_arxiv"
    description: str = "Gets the abstract from an ArXiv paper given the arxiv ID."
    args_schema: type[ArxivInput] = ArxivInput

    def _run(self, arxiv_id: str) -> str:
        try:
            # Clean the arxiv_id
            arxiv_id = arxiv_id.strip().replace('"', '').replace("'", '')
            
            # Get paper page in HTML
            res = requests.get(
                f"https://export.arxiv.org/abs/{arxiv_id}",
                headers={'User-Agent': 'Mozilla/5.0 (Research Assistant Bot)'}
            )
            res.raise_for_status()
            
            # Updated regex pattern to match the actual HTML structure
            abstract_pattern = re.compile(
                r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>(.*?)</blockquote>',
                re.DOTALL
            )
            
            # Search HTML for abstract
            re_match = abstract_pattern.search(res.text)
            if not re_match:
                return f"Error: Could not find abstract for ArXiv ID {arxiv_id}"
            
            # Clean up the abstract text
            abstract = re_match.group(1)
            abstract = re.sub(r'<[^>]+>', ' ', abstract)  # Remove HTML tags
            abstract = re.sub(r'\s+', ' ', abstract)      # Normalize whitespace
            abstract = abstract.strip()                   # Remove leading/trailing whitespace
            
            return abstract
            
        except requests.RequestException as e:
            return f"Error fetching ArXiv paper {arxiv_id}: {str(e)}"
        except Exception as e:
            return f"Error processing ArXiv paper {arxiv_id}: {str(e)}"

    async def _arun(self, arxiv_id: str) -> str:
        return self._run(arxiv_id)

@tool
def Search(queries: List[str]):
    """A list of one or more search queries to find good resources to support the research."""

@tool
def WriteReport(report: str):
    """Write the research report."""

@tool
def WriteResearchQuestion(research_question: str):
    """Write the research question."""

@tool
def DeleteResources(urls: List[str]):
    """Delete the URLs from the resources."""

async def chat_node(state: AgentState, config: RunnableConfig):
    """Chat Node"""

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "report",
            "tool": "WriteReport",
            "tool_argument": "report",
        }, {
            "state_key": "research_question",
            "tool": "WriteResearchQuestion",
            "tool_argument": "research_question",
        }],
        emit_tool_calls="DeleteResources"
    )

    # Initialize state values
    resources: List[Resource] = state.get("resources", [])
    logs: List[Log] = state.get("logs", [])
    research_question: str = state.get("research_question", "")
    report: str = state.get("report", "")

    # Process ArXiv ID if present in last message
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        message = state["messages"][-1].content
        arxiv_match = re.search(r'arxiv_id\s*=\s*["\']?([0-9.]+)["\']?', message)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            # Add log for ArXiv fetch
            logs.append({
                "message": f"Fetching ArXiv paper {arxiv_id}",
                "done": False
            })
            
            fetch_tool = FetchArxivTool()
            abstract = await fetch_tool._arun(arxiv_id)
            
            # Update log status
            logs[-1]["done"] = True
            
            # Add paper as a resource
            new_resource: Resource = {
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "title": f"ArXiv Paper {arxiv_id}",
                "description": abstract[:200] + "..." if len(abstract) > 200 else abstract
            }
            resources.append(new_resource)
            
            # Update report
            new_report = report
            if new_report:
                new_report += "\n\n"
            new_report += f"ArXiv Paper Analysis (ID: {arxiv_id}):\n\nAbstract:\n{abstract}\n"
            
            return {
                "report": new_report,
                "resources": resources,
                "logs": logs,
                "messages": state["messages"] + [AIMessage(content=f"Retrieved and analyzed ArXiv paper {arxiv_id}")]
            }

    # Process existing resources
    processed_resources = []
    for resource in resources:
        content = get_resource(resource["url"])
        if content == "ERROR":
            continue
        processed_resources.append({
            **resource,
            "content": content
        })

    model = get_model(state)
    ainvoke_kwargs = {}
    if model.__class__.__name__ in ["ChatOpenAI"]:
        ainvoke_kwargs["parallel_tool_calls"] = False

    response = await model.bind_tools(
        [
            Search,
            WriteReport,
            WriteResearchQuestion,
            DeleteResources,
            FetchArxivTool(),
        ],
        **ainvoke_kwargs
    ).ainvoke([
        SystemMessage(
            content=f"""
            You are a research assistant. You help the user with writing a research report.
            
            Important instructions for ArXiv papers:
            - When a user provides an ArXiv ID, use the fetch_arxiv tool to get the abstract
            - After fetching, analyze the paper and incorporate findings into the report
            
            General instructions:
            - Use the search tool for web searches
            - Use resources to answer questions but don't recite them directly
            - Use WriteReport tool for all reports
            - If a research question exists, don't ask for it again

            Current research question: {research_question}
            Current report: {report}
            Available resources: {processed_resources}
            """
        ),
        *state["messages"],
    ], config)

    ai_message = cast(AIMessage, response)

    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        if tool_call["name"] == "WriteReport":
            return {
                "report": tool_call["args"].get("report", ""),
                "resources": resources,
                "logs": logs,
                "messages": [ai_message, ToolMessage(tool_call_id=tool_call["id"], content="Report written.")]
            }
        elif tool_call["name"] == "WriteResearchQuestion":
            return {
                "research_question": tool_call["args"]["research_question"],
                "resources": resources,
                "logs": logs,
                "messages": [ai_message, ToolMessage(tool_call_id=tool_call["id"], content="Research question written.")]
            }

    return {
        "messages": [ai_message],
        "resources": resources,
        "logs": logs
    }