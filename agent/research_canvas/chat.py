
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
from tavily import TavilyClient
import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "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"))

class ArxivSearchInput(BaseModel):
    """Input for ArXiv search."""
    query: str = Field(description="The search query for finding relevant papers")
    max_results: int = Field(default=5, description="Maximum number of papers to return")

class SearchInput(BaseModel):
    """Input for web search."""
    queries: List[str] = Field(description="List of search queries")
    max_results: int = Field(default=5, description="Maximum number of results per query")

class ArxivSearchTool(BaseTool):
    name: str = "search_arxiv"
    description: str = "Searches ArXiv for relevant papers based on a query. Use this tool only when specifically asked about arxiv papers or given an arxiv ID."
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

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Performs a web search and returns structured results using Tavily."
    args_schema: type[SearchInput] = SearchInput

    def _run(self, queries: List[str], max_results: int = 5) -> str:
        try:
            all_results = []
            
            for query in queries:
                # Use Tavily search for each query
                response = tavily_client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced"
                )
                
                # Process results
                for result in response['results']:
                    all_results.append({
                        "title": result['title'],
                        "url": result['url'],
                        "summary": result['content']
                    })
                    
                    # Break if we have enough results
                    if len(all_results) >= max_results:
                        break
            
            return json.dumps({
                "results": all_results[:max_results],
                "total": len(all_results[:max_results])
            })
        except Exception as e:
            return json.dumps({"error": f"Error performing web search: {str(e)}"})

    async def _arun(self, queries: List[str], max_results: int = 5) -> str:
        return self._run(queries, max_results)

@tool
def WriteReport(report: str): # pylint: disable=invalid-name,unused-argument
    """Write the research report."""

@tool
def WriteResearchQuestion(research_question: str): # pylint: disable=invalid-name,unused-argument
    """Write the research question."""

@tool
def DeleteResources(urls: List[str]): # pylint: disable=invalid-name,unused-argument
    """Delete the URLs from the resources. If empty list is provided, delete all resources."""

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

    # Initialize or reset state based on new conversation
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        # Check if this is a new conversation by looking at the message content
        message = state["messages"][-1].content.lower()
        
        # Reset conditions - add more as needed
        should_reset = any([
            "new research" in message,
            "start over" in message,
            "new topic" in message,
            # If this is the first message in a conversation
            len(state["messages"]) == 1
        ])
        
        if should_reset:
            state["resources"] = []
            state["report"] = ""
            state["research_question"] = ""
            return {
                "resources": [],
                "report": "",
                "research_question": "",
                "messages": [AIMessage(content="""
=================================== 
AGENT: State Manager
ACTION: Reset State
===================================

Starting new research conversation. Previous draft and resources have been cleared.""")]
            }

    state["resources"] = state.get("resources", [])
    research_question = state.get("research_question", "")
    report = state.get("report", "")

    # Check for delete command
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        message = state["messages"][-1].content.lower()
        if "delete" in message and ("everything" in message or "all" in message):
            state["resources"] = []
            state["report"] = ""  # Also clear the report when deleting everything
            return {
                "resources": [],
                "report": "",
                "messages": [AIMessage(content="""
=================================== 
AGENT: Resource Manager
ACTION: Delete All Resources 
===================================

All resources and report have been deleted.""")]
            }

    # Check for ArXiv query
    is_arxiv_query = False
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        message = state["messages"][-1].content.lower()
        is_arxiv_query = any([
            'arxiv' in message,
            'arxiv_id' in message,
            'arxiv id' in message,
            re.search(r'\d{4}\.\d{4,5}', message) is not None
        ])

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

    system_message = f"""
You are a research assistant that helps users find information and write reports.

Important instructions:
Agent Identification:
- At the start of EVERY response, indicate which agent you are using:
  "Agent: ArXiv Search" or "Agent: Web Search"
{'- Use the search_arxiv tool for this query.' if is_arxiv_query else '- Use the web_search tool for this query.'}
- Use WriteReport tool to update the report with new findings
- Always indicate the source of information in your response
- If a research question exists, do not ask for it again

Report Structure:
# Research Report

## Key Points
- Main discoveries and findings
- Critical insights from papers
- Important metrics and results

## Main Body

### Background
- Research context and significance
- Current state of the field
- Research gaps identified

### Methodology
- Research approaches used
- Data collection methods
- Analysis techniques

### Results and Analysis
- Synthesis of key findings
- Comparison across papers
- Statistical significance
- Key implications

## Summary and Conclusions
- Major findings synthesis
- Research implications
- Future research directions
- Practical applications

Research question: {research_question}
Current report: {report}
Available resources: {resources}

Format Guidelines:
- Use markdown headers (##) for sections
- Use bullet points (*) for lists
- Include citations [Author, Year]
- Use blockquotes (>) for important quotes
"""

    response = await model.bind_tools(
        [
            WebSearchTool(),
            WriteReport,
            WriteResearchQuestion,
            DeleteResources,
            ArxivSearchTool(),
        ],
        **ainvoke_kwargs
    ).ainvoke([
        SystemMessage(content=system_message),
        *state["messages"],
    ], config)

    ai_message = cast(AIMessage, response)

    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        
        if tool_call["name"] == "search_arxiv":
            result = json.loads(await ArxivSearchTool()._arun(**tool_call["args"]))
            
            if "error" in result:
                return {
                    "messages": [ai_message, ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"""
===================================
AGENT: ArXiv Search
STATUS: Error
===================================

{result['error']}"""
                    )]
                }
            
            papers = result["papers"]
            urls = []
            for paper in papers:
                url = f"https://arxiv.org/abs/{paper['arxiv_id']}"
                urls.append(url)
                new_resource = {
                    "url": url,
                    "title": paper["title"],
                    "description": paper["summary"]
                }
                state["resources"].append(new_resource)
            
            # Replace the report instead of appending
            new_report = f"ArXiv Paper Analysis:\n\n"
            for paper in papers:
                new_report += f"Title: {paper['title']}\n"
                new_report += f"ArXiv ID: {paper['arxiv_id']}\n"
                new_report += f"Abstract:\n{paper['summary']}\n\n"
            
            return {
                "report": new_report,
                "resources": state["resources"],
                "messages": [ai_message, ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"""
===================================
AGENT: ArXiv Search
SOURCES: {', '.join(urls)}
===================================

Found {len(papers)} papers. Updated resources and report."""
                )]
            }
            
        elif tool_call["name"] == "web_search":
            result = json.loads(await WebSearchTool()._arun(**tool_call["args"]))
            
            if "error" in result:
                return {
                    "messages": [ai_message, ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"""
===================================
AGENT: Web Search
STATUS: Error
===================================

{result['error']}"""
                    )]
                }
            
            search_results = result["results"]
            urls = []
            
            # Replace the report instead of appending
            new_report = f"Web Search Results:\n\n"
            
            for result in search_results:
                urls.append(result["url"])
                new_resource = {
                    "url": result["url"],
                    "title": result["title"],
                    "description": result["summary"]
                }
                state["resources"].append(new_resource)
                
                new_report += f"Title: {result['title']}\n"
                new_report += f"URL: {result['url']}\n"
                new_report += f"Summary: {result['summary']}\n\n"
            
            return {
                "report": new_report,
                "resources": state["resources"],
                "messages": [ai_message, ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"""
===================================
AGENT: Web Search
SOURCES: {', '.join(urls)}
===================================

Found {len(search_results)} relevant resources. Updated report."""
                )]
            }
            
        elif tool_call["name"] == "WriteReport":
            return {
                "report": tool_call["args"].get("report", ""),  # Replace entire report
                "messages": [ai_message, ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="""
===================================
AGENT: Report Writer
ACTION: Update Report
===================================

Report has been updated."""
                )]
            }
            
        elif tool_call["name"] == "WriteResearchQuestion":
            return {
                "research_question": tool_call["args"]["research_question"],
                "messages": [ai_message, ToolMessage(
                    tool_call_id=tool_call["id"],
                    content="""
===================================
AGENT: Research Question Writer
ACTION: Set Question
===================================

Research question has been written."""
                )]
            }
            
        elif tool_call["name"] == "DeleteResources":
            urls_to_delete = tool_call["args"].get("urls", [])
            if not urls_to_delete:
                state["resources"] = []
                state["report"] = ""  # Also clear the report when deleting everything
                return {
                    "resources": [],
                    "report": "",
                    "messages": [ai_message, ToolMessage(
                        tool_call_id=tool_call["id"],
                        content="""
===================================
AGENT: Resource Manager
ACTION: Delete All Resources
===================================

All resources and report have been deleted."""
                    )]
                }
            else:
                state["resources"] = [r for r in state["resources"] if r["url"] not in urls_to_delete]
                return {
                    "resources": state["resources"],
                    "messages": [ai_message, ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"""
===================================
AGENT: Resource Manager
ACTION: Delete Specific Resources
===================================

{len(urls_to_delete)} resources have been deleted."""
                    )]
                }

    if isinstance(response, AIMessage):
        sources_used = [r["url"] for r in resources if hasattr(r, "url")]
        header = f"""
===================================
AGENT: {"ArXiv Search" if is_arxiv_query else "Web Search"}
SOURCES: {', '.join(sources_used) if sources_used else 'None'}
===================================

"""
        response.content = header + response.content

    return {
        "messages": response
    }