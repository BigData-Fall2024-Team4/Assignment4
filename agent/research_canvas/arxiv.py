"""ArXiv Agent Node"""

from typing import List, Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from copilotkit.langchain import copilotkit_customize_config
from research_canvas.state import AgentState
from research_canvas.model import get_model
import requests
import re

class ArxivSearchInput(BaseModel):
    """Input for ArXiv search."""
    query: str = Field(description="The search query for finding relevant papers")
    max_results: int = Field(default=5, description="Maximum number of papers to return")

class ArxivSearchTool(BaseTool):
    name: str = "search_arxiv"
    description: str = "Searches ArXiv for relevant papers based on a query."
    args_schema: type[ArxivSearchInput] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
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
                    paper = {
                        "arxiv_id": id_match.group(1),
                        "title": re.sub(r'\s+', ' ', title_match.group(1)).strip(),
                        "summary": re.sub(r'\s+', ' ', summary_match.group(1)).strip()
                    }
                    papers.append(paper)
            
            return papers
            
        except requests.RequestException as e:
            return [{"error": f"Error searching ArXiv: {str(e)}"}]
        except Exception as e:
            return [{"error": f"Error processing ArXiv search: {str(e)}"}]

    async def _arun(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        return self._run(query, max_results)

@tool
def WriteReport(report: str):
    """Write the research report."""

async def arxiv_agent_node(state: AgentState, config: RunnableConfig):
    """ArXiv Agent Node"""
    
    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "report",
            "tool": "WriteReport",
            "tool_argument": "report",
        }]
    )

    # Initialize state values
    resources = state.get("resources", [])
    logs = state.get("logs", [])
    report = state.get("report", "")
    research_question = state.get("research_question", "")

    # Process search query if present in last message
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        message = state["messages"][-1].content
        if "search_papers" in message.lower():
            # Extract search query
            query = message.lower().replace("search_papers", "").strip()
            
            # Add log for ArXiv search
            logs.append({
                "message": f"Searching ArXiv for: {query}",
                "done": False
            })
            
            # Perform search
            search_tool = ArxivSearchTool()
            papers = await search_tool._arun(query)
            
            # Update log status
            logs[-1]["done"] = True
            
            # Process results and update resources
            for paper in papers:
                if "error" not in paper:
                    new_resource = {
                        "url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
                        "title": paper["title"],
                        "description": paper["summary"][:200] + "..." if len(paper["summary"]) > 200 else paper["summary"]
                    }
                    resources.append(new_resource)
            
            # Update report with search results
            new_report = report
            if new_report:
                new_report += "\n\n"
            new_report += f"ArXiv Search Results for '{query}':\n\n"
            for paper in papers:
                if "error" not in paper:
                    new_report += f"Title: {paper['title']}\n"
                    new_report += f"ID: {paper['arxiv_id']}\n"
                    new_report += f"Summary: {paper['summary']}\n\n"
            
            return {
                "report": new_report,
                "resources": resources,
                "logs": logs,
                "messages": state["messages"] + [AIMessage(content=f"Found {len(papers)} relevant papers on ArXiv")]
            }

    model = get_model(state)
    response = await model.bind_tools(
        [
            ArxivSearchTool(),
            WriteReport,
        ]
    ).ainvoke([
        SystemMessage(
            content=f"""
            You are an ArXiv research assistant. You help users find and analyze relevant research papers.
            
            Instructions:
            - Use the search_arxiv tool to find relevant papers based on the research question
            - Analyze papers and incorporate findings into the report
            - Use WriteReport tool to update the research report
            - Provide concise summaries and key findings
            
            Current research question: {research_question}
            Current report: {report}
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
                "messages": [ai_message, ToolMessage(tool_call_id=tool_call["id"], content="Report updated with paper analysis.")]
            }

    return {
        "messages": [ai_message],
        "resources": resources,
        "logs": logs
    }