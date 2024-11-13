"""Chat Node"""

from typing import List, cast, Dict
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from copilotkit.langchain import copilotkit_customize_config
from research_canvas.state import AgentState
from research_canvas.model import get_model
from research_canvas.download import get_resource
import requests
import re
import json

class ArxivSearchInput(BaseModel):
    """Input for ArXiv search."""
    query: str = Field(description="The search query for finding relevant papers")
    max_results: int = Field(default=5, description="Maximum number of papers to return")

class ArxivSearchTool(BaseTool):
    name: str = "search_arxiv"
    description: str = "Searches ArXiv for relevant papers based on a query. Use this tool for finding academic papers."
    args_schema: type[ArxivSearchInput] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            query = query.strip().replace(' ', '+')
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
            response = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Research Assistant Bot)'}
            )
            response.raise_for_status()
            
            papers = []
            entries = re.finditer(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
            
            for entry in entries:
                entry_text = entry.group(1)
                title_match = re.search(r'<title>(.*?)</title>', entry_text, re.DOTALL)
                id_match = re.search(r'<id>http://arxiv.org/abs/(.*?)</id>', entry_text)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry_text, re.DOTALL)
                
                if title_match and id_match and summary_match:
                    papers.append({
                        "arxiv_id": id_match.group(1),
                        "title": re.sub(r'\s+', ' ', title_match.group(1)).strip(),
                        "summary": re.sub(r'\s+', ' ', summary_match.group(1)).strip()
                    })
            
            # Return structured data as a JSON string
            return json.dumps({
                "papers": papers,
                "total": len(papers)
            })
            
        except requests.RequestException as e:
            return json.dumps({"error": f"Error searching ArXiv: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Error processing ArXiv search: {str(e)}"})

    async def _arun(self, query: str, max_results: int = 5) -> str:
        return self._run(query, max_results)

@tool
def WriteReport(report: str): # pylint: disable=invalid-name,unused-argument
    """Write the research report."""

@tool
def WriteResearchQuestion(research_question: str): # pylint: disable=invalid-name,unused-argument
    """Write the research question."""

@tool
def DeleteResources(urls: List[str]): # pylint: disable=invalid-name,unused-argument
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

    state["resources"] = state.get("resources", [])
    research_question = state.get("research_question", "")
    report = state.get("report", "")

    resources = []
    for resource in state["resources"]:
        content = get_resource(resource["url"])
        if content == "ERROR":
            continue
        resources.append({
            **resource,
            "content": content
        })

    model = get_model(state)
    ainvoke_kwargs = {}
    if model.__class__.__name__ in ["ChatOpenAI"]:
        ainvoke_kwargs["parallel_tool_calls"] = False

    response = await model.bind_tools(
        [
            WriteReport,
            WriteResearchQuestion,
            DeleteResources,
            ArxivSearchTool(),
        ],
        **ainvoke_kwargs
    ).ainvoke([
        SystemMessage(
            content=f"""
            You are a research assistant specializing in scientific literature. You help users find and analyze research papers.
            
            Important instructions:
            - When the user asks about papers or research, ALWAYS use the search_arxiv tool to find relevant papers
            - After finding papers, analyze them and incorporate the findings into the report using the WriteReport tool
            - Never just summarize papers without adding them to resources and report
            - If a research question exists, do not ask for it again
            
            Research question: {research_question}
            Current report: {report}
            Available resources: {resources}
            """
        ),
        *state["messages"],
    ], config)

    ai_message = cast(AIMessage, response)

    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        
        if tool_call["name"] == "search_arxiv":
            # Process ArXiv search results
            result = json.loads(await ArxivSearchTool()._arun(**tool_call["args"]))
            
            if "error" in result:
                return {
                    "messages": [
                        ai_message,
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=result["error"]
                        )
                    ]
                }
            
            # Update resources and report
            papers = result["papers"]
            for paper in papers:
                new_resource = {
                    "url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
                    "title": paper["title"],
                    "description": paper["summary"]
                }
                state["resources"].append(new_resource)
            
            new_report = report
            if new_report:
                new_report += "\n\n"
            new_report += f"ArXiv Paper Analysis:\n\n"
            for paper in papers:
                new_report += f"Title: {paper['title']}\n"
                new_report += f"ArXiv ID: {paper['arxiv_id']}\n"
                new_report += f"Abstract:\n{paper['summary']}\n\n"
            
            return {
                "report": new_report,
                "resources": state["resources"],
                "messages": [
                    ai_message,
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"Found {len(papers)} papers from ArXiv. Papers have been added to resources and report."
                    )
                ]
            }
            
        elif tool_call["name"] == "WriteReport":
            return {
                "report": tool_call["args"].get("report", ""),
                "messages": [ai_message, ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Report written."
                )]
            }
            
        elif tool_call["name"] == "WriteResearchQuestion":
            return {
                "research_question": tool_call["args"]["research_question"],
                "messages": [ai_message, ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="Research question written."
                )]
            }

    return {
        "messages": response
    }