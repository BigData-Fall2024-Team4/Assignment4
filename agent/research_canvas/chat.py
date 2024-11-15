
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





from datetime import datetime
import os
from fpdf import FPDF
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import re

class SavePDFInput(BaseModel):
    """Input for PDF saving."""
    report: str = Field(description="The report content to save as PDF")

class ResearchPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.left_margin = 25
        self.right_margin = 25
        self.top_margin = 25
        self.set_margins(self.left_margin, self.top_margin, self.right_margin)
        self.set_auto_page_break(True, margin=25)
        
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Research Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def clean_text(self, text):
        """Clean text of problematic characters and URLs."""
        # Remove URLs and replace with cleaner format
        text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1', text)
        # Convert to ASCII
        text = text.encode('ascii', 'replace').decode('ascii')
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text

    def clean_url(self, url):
        """Clean URL text."""
        # Remove any problematic characters from URLs
        return url.strip().replace(' ', '%20')

    def write_title(self, title):
        """Write a section title."""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, self.clean_text(title), 0, 1, 'L')
        self.ln(5)

    def write_subtitle(self, subtitle):
        """Write a subsection title."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, self.clean_text(subtitle), 0, 1, 'L')
        self.ln(2)

    def write_paragraph(self, text):
        """Write a paragraph of text."""
        self.set_font('Arial', '', 11)
        cleaned_text = self.clean_text(text)
        effective_width = self.w - self.left_margin - self.right_margin
        self.multi_cell(effective_width, 6, cleaned_text)
        self.ln(4)

    def write_bullet_point(self, text):
        """Write a bullet point."""
        self.set_font('Arial', '', 11)
        effective_width = self.w - self.left_margin - self.right_margin - 10
        x_start = self.get_x()
        self.cell(5, 6, '-', 0, 0, 'L')
        self.set_x(x_start + 8)
        cleaned_text = self.clean_text(text)
        self.multi_cell(effective_width, 6, cleaned_text)
        self.ln(2)

    def write_link(self, title, url):
        """Write a link reference."""
        self.set_font('Arial', '', 10)
        effective_width = self.w - self.left_margin - self.right_margin
        cleaned_title = self.clean_text(title)
        cleaned_url = self.clean_url(url)
        self.multi_cell(effective_width, 6, f"{cleaned_title}\n{cleaned_url}")
        self.ln(2)

class SavePDFTool(BaseTool):
    name: str = "save_report_pdf"
    description: str = "Saves the current research report as a PDF file"
    args_schema: Type[BaseModel] = SavePDFInput
    
    def _run(self, report: str) -> str:
        try:
            print("Starting PDF generation...")
            
            if not os.path.exists('reports'):
                os.makedirs('reports')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'reports/research_report_{timestamp}.pdf'
            
            pdf = ResearchPDF()
            pdf.add_page()
            
            # Process report sections
            sections = report.split('\n\n')
            in_bullet_list = False
            
            for section in sections:
                if not section.strip():
                    continue
                
                lines = section.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Handle different line types
                        if line.startswith('# '):
                            pdf.write_title(line[2:])
                            in_bullet_list = False
                        elif line.startswith('## '):
                            pdf.write_subtitle(line[3:])
                            in_bullet_list = False
                        elif line.startswith('### '):
                            pdf.write_subtitle(line[4:])
                            in_bullet_list = False
                        elif line.startswith('- ') or line.startswith('* '):
                            pdf.write_bullet_point(line[2:])
                            in_bullet_list = True
                        elif '[' in line and '](' in line and ')' in line:
                            # Handle links
                            match = re.match(r'\[(.*?)\]\((.*?)\)', line)
                            if match:
                                title, url = match.groups()
                                pdf.write_link(title, url)
                            in_bullet_list = False
                        else:
                            if in_bullet_list:
                                pdf.write_bullet_point(line)
                            else:
                                pdf.write_paragraph(line)
                    except Exception as e:
                        print(f"Error processing line '{line[:50]}...': {str(e)}")
                        continue
            
            pdf.output(filename)
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                return f"Report successfully saved as {filename} (Size: {file_size:,} bytes)"
            else:
                return "Error: PDF file was not created successfully"
            
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error saving PDF: {str(e)}"

    async def _arun(self, report: str) -> str:
        """Async implementation that calls the sync version."""
        return self._run(report)




import os
import json
from datetime import datetime
from pathlib import Path

class SaveCodelabTool:
    """Tool for saving chat interactions and reports as a codelab in markdown format and exporting with claat"""
    
    def __init__(self):
        self.name = "save_codelab"
        self.description = "Saves the chat interaction and report as a codelab markdown file and exports it"
        self.base_dir = 'codelabs'
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _format_messages(self, messages, report=None):
        """Format messages into markdown codelab structure"""
        # Original metadata format
        codelab_content = """summary: Interactive CodeLab
id: interactive-codelab
categories: codelab
environments: web
status: Published
feedback link: https://github.com/googlecodelabs/tools
analytics account: UA-XXXXXXXX-1
authors: Generated
duration: 10

# Interactive CodeLab

"""
        
        # Add duration steps
        step_number = 0

        # Add report if available
        if report:
            step_number += 1
            codelab_content += f"## Step {step_number}: Research Report\n\n"
            codelab_content += f"Duration: 5:00\n\n"
            codelab_content += report + "\n\n"
            codelab_content += "Positive\n: ✅ Research findings reviewed\n\n"
            codelab_content += "Negative\n: ❌ Additional research may be needed\n\n"

        # Format conversation into steps
        current_step = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                step_number += 1
                current_step = f"## Step {step_number}: User Query\n\nDuration: 5:00\n\n"
                current_step += f"{msg.content}\n\n"
            elif isinstance(msg, AIMessage):
                if current_step:
                    current_step += "### Response\n\n"
                    content = msg.content
                    
                    # Remove agent header if present
                    if "===================================" in content:
                        content = content.split("===================================")[-1].strip()
                    
                    current_step += f"{content}\n\n"
                    
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        current_step += "### Tool Output\n\n"
                        for tool_call in msg.tool_calls:
                            current_step += f"```python\n{tool_call['name']}: {json.dumps(tool_call['args'], indent=2)}\n```\n\n"
                    
                    codelab_content += current_step
                    current_step = None
            elif isinstance(msg, ToolMessage):
                if current_step:
                    current_step += "### Tool Message\n\n"
                    current_step += f"```\n{msg.content}\n```\n\n"

        # Add final step with summary
        step_number += 1
        codelab_content += f"""## Step {step_number}: Summary and Next Steps

Duration: 5:00

### Key Takeaways

* Review the research findings above
* Consider the conversation history
* Examine tool interactions and outputs

Positive
: ✅ You've completed this interactive research codelab
: ✅ You can now apply these insights to your work

Negative
: ❌ Remember to verify all findings independently
: ❌ Consider seeking additional sources for validation

### Next Steps

1. Review all research findings
2. Validate key information
3. Apply insights to your project

"""

        return codelab_content

    async def _arun(self, messages, state=None, **kwargs):
        """Save markdown file and run claat export"""
        try:
            # Get report from state if available
            report = state.get("report", "") if state else ""
            
            # Create markdown file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            md_filename = f"codelab_{timestamp}.md"
            md_path = os.path.join(self.base_dir, md_filename)
            
            # Save markdown content with report
            content = self._format_messages(messages, report)
            
            # Debug: Print content structure
            print(f"Generated content structure:\n{content[:500]}...")
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Created markdown file: {md_path}")
            
            # Run claat export command
            try:
                result = subprocess.run(
                    ['claat', 'export', md_path],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )
                
                if result.returncode == 0:
                    codelab_dir = os.path.join(self.base_dir, md_filename[:-3])
                    
                    if os.path.exists(codelab_dir):
                        return f"""
===================================
AGENT: Codelab Generator
STATUS: Success
===================================

1. Markdown file created: {md_path}
2. Codelab exported successfully: {codelab_dir}
3. CLAAT Output: {result.stdout}

Content structure:
- Original metadata preserved
- Report included as Step 1 (if provided)
- Each conversation part as a numbered step
- Summary as final step

You can find:
- Markdown file at: {md_path}
- Generated codelab at: {codelab_dir}

To verify format:
1. Check {md_path} content
2. Ensure metadata section matches original
3. Verify step numbering and duration tags"""
                    else:
                        return f"""
===================================
AGENT: Codelab Generator
STATUS: Partial Success
===================================

Markdown file created: {md_path}

CLAAT Export failed to create directory
STDOUT: {result.stdout}
STDERR: {result.stderr}
Return Code: {result.returncode}

To verify markdown format:
cat {md_path}

Please run manually:
cd {self.base_dir}
claat export {md_filename}"""
                else:
                    return f"""
===================================
AGENT: Codelab Generator
STATUS: Partial Success
===================================

Markdown file created: {md_path}

CLAAT Export failed
STDOUT: {result.stdout}
STDERR: {result.stderr}
Return Code: {result.returncode}

To verify markdown format:
cat {md_path}

Please verify claat installation and run manually:
cd {self.base_dir}
claat export {md_filename}"""
                    
            except FileNotFoundError:
                return f"""
===================================
AGENT: Codelab Generator
STATUS: Partial Success
===================================

Markdown file created: {md_path}

CLAAT not found in PATH. Please:
1. Install claat: go install github.com/googlecodelabs/tools/claat@latest
2. Add to PATH: export PATH=$PATH:$(go env GOPATH)/bin
3. Run manually:
   cd {self.base_dir}
   claat export {md_filename}"""
                
            except Exception as e:
                return f"""
===================================
AGENT: Codelab Generator
STATUS: Partial Success
===================================

Markdown file created: {md_path}

CLAAT Export error: {str(e)}

To verify markdown format:
cat {md_path}

Please run manually:
cd {self.base_dir}
claat export {md_filename}"""
                
        except Exception as e:
            return f"""
===================================
AGENT: Codelab Generator
STATUS: Error
===================================

Failed to create markdown file: {str(e)}

Working Directory: {os.getcwd()}
Target Directory: {self.base_dir}
Directory exists: {os.path.exists(self.base_dir)}"""


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
    """Chat Node with improved file handling and debugging"""
    
    # Initialize PDF saving tool
    pdf_tool = SavePDFTool()
    
    # Create base directories with more permissive permissions
    BASE_DIRS = ['chat_outputs', 'reports']
    for directory in BASE_DIRS:
        try:
            # Make directory with full permissions (777)
            os.makedirs(directory, mode=0o777, exist_ok=True)
            os.chmod(directory, 0o777)
            print(f"Successfully created/verified directory {directory} with permissions 777")
        except Exception as e:
            print(f"Warning: Failed to create/set permissions for {directory}: {str(e)}")
            # Try to get current permissions if directory exists
            try:
                current_perms = oct(os.stat(directory).st_mode)[-3:]
                print(f"Current permissions for {directory}: {current_perms}")
            except:
                print(f"Could not read permissions for {directory}")

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
        emit_tool_calls=["DeleteResources", "save_report_pdf"]
    )

    # Check for save commands early in the function
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        message = state["messages"][-1].content.lower()
        
        # Handle markdown save command
        if "save" in message and "md" in message:
            try:
                # Print current working directory and verify chat_outputs exists
                cwd = os.getcwd()
                print(f"Current working directory: {cwd}")
                print(f"chat_outputs exists: {os.path.exists('chat_outputs')}")
                
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join("chat_outputs", f"chat_{timestamp}.md")
                
                # Format messages into markdown
                md_content = "# Chat History\n\n"
                
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage):
                        md_content += "## Question\n\n"
                        md_content += f"{msg.content}\n\n"
                    elif isinstance(msg, AIMessage):
                        md_content += "## Response\n\n"
                        md_content += f"{msg.content}\n\n"
                    elif isinstance(msg, ToolMessage):
                        md_content += "### Tool Output\n\n"
                        md_content += f"```\n{msg.content}\n```\n\n"

                # Print debug info before saving
                print(f"Attempting to save to: {filename}")
                
                # Create parent directory again just to be sure
                os.makedirs("chat_outputs", mode=0o777, exist_ok=True)
                
                # Save to file with explicit encoding and full permissions
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(md_content)
                
                # Set file permissions to be fully readable/writable
                os.chmod(filename, 0o666)
                
                # Get absolute path and verify file was created
                abs_path = os.path.abspath(filename)
                file_exists = os.path.exists(filename)
                
                print(f"File saved successfully: {file_exists}")
                print(f"Absolute path: {abs_path}")
                
                return {
                    "messages": [AIMessage(content=f"""
===================================
AGENT: Markdown Generator
ACTION: Save Chat History
STATUS: Success
===================================

Chat history has been saved to: {abs_path}
File exists: {file_exists}
Directory permissions: {oct(os.stat('chat_outputs').st_mode)[-3:]}
File permissions: {oct(os.stat(filename).st_mode)[-3:]}
Working directory: {cwd}""")]
                }
            except Exception as e:
                # Enhanced error reporting
                cwd = os.getcwd()
                error_details = []
                
                try:
                    error_details.append(f"Working Directory: {cwd}")
                    error_details.append(f"Directory exists: {os.path.exists('chat_outputs')}")
                    if os.path.exists('chat_outputs'):
                        error_details.append(f"Directory permissions: {oct(os.stat('chat_outputs').st_mode)[-3:]}")
                    error_details.append(f"Python process user: {os.getuid()}")
                    error_details.append(f"Python process group: {os.getgid()}")
                except Exception as debug_e:
                    error_details.append(f"Debug error: {str(debug_e)}")
                
                return {
                    "messages": [AIMessage(content=f"""
===================================
AGENT: Markdown Generator
ACTION: Save Chat History
STATUS: Error
===================================

Failed to save chat history: {str(e)}

Debug Information:
{chr(10).join(error_details)}

Error Type: {type(e).__name__}
Full Error: {str(e)}

Please ensure:
1. The application has write permissions in the current directory
2. The chat_outputs directory exists and is writable
3. There is sufficient disk space
4. The path is accessible from the current working directory

Contact support if the issue persists.""")]
                }

        # Handle PDF save command
        elif "save" in message and "pdf" in message:
            report = state.get("report", "")
            if report:
                try:
                    result = pdf_tool._run(report)
                    print(f"PDF Generation Result: {result}")
                    return {
                        "messages": [AIMessage(content=f"""
===================================
AGENT: PDF Generator
ACTION: Save Report
STATUS: Success
===================================

{result}

The PDF has been saved in the 'reports' directory.""")]
                    }
                except Exception as e:
                    print(f"PDF Generation Error: {str(e)}")
                    return {
                        "messages": [AIMessage(content=f"""
===================================
AGENT: PDF Generator
ACTION: Save Report
STATUS: Error
===================================

Failed to generate PDF: {str(e)}

Please try again or contact support if the issue persists.""")]
                    }
            else:
                return {
                    "messages": [AIMessage(content="""
===================================
AGENT: PDF Generator
STATUS: Error
===================================

No report content available to save as PDF. Please generate a report first by searching for some information.""")]
                }
        
        # Reset conditions
        should_reset = any([
            "new research" in message,
            "start over" in message,
            "new topic" in message,
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
            state["report"] = ""
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
- When citing sources, use the format [Title](URL)
- Include relevant quotes and summaries from sources
- If a research question exists, do not ask for it again

Report Structure:
# Research Report

### Key Points
- Main discoveries and findings from sources
- Critical insights with source citations
- Important metrics and results

### Sources Overview
{chr(10).join([f'- [{r["title"]}]({r["url"]})' for r in resources]) if resources else '- No sources yet'}

### Main Body

### Background
- Research context with source citations
- Current state of the field
- Research gaps identified

### Methodology
- Research approaches from sources
- Data collection methods
- Analysis techniques

### Results and Analysis
- Synthesis of key findings
- Comparison across sources
- Statistical significance
- Key implications

### Source Details
{chr(10).join([f'### {r["title"]}\n- URL: {r["url"]}\n- Summary: {r["description"]}\n' for r in resources]) if resources else '- No sources yet'}

### Summary and Conclusions
- Major findings synthesis with citations
- Research implications
- Future research directions
- Practical applications

Format Guidelines:
- Use markdown headers (##) for sections
- Use bullet points (*) for lists
- Include citations [Author URL]
- Use blockquotes (>) for important quotes
- Always link to sources when citing them

Research question: {research_question}
Current report: {report}
Available resources: {resources}
"""

    response = await model.bind_tools(
        [
            WebSearchTool(),
            WriteReport,
            WriteResearchQuestion,
            DeleteResources,
            ArxivSearchTool(),
            pdf_tool,
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
            
            new_report = "# Web Search Results Analysis\n\n"
            new_report += "## Found Resources\n\n"
            
            for result in search_results:
                urls.append(result["url"])
                new_resource = {
                    "url": result["url"],
                    "title": result["title"],
                    "description": result["summary"]
                }
                state["resources"].append(new_resource)
                
                new_report += f"Title: {result['title']}\n"
                new_report += f"**Source**: [{result['url']}]({result['url']})\n\n"
                new_report += f"URL: {result['url']}\n"
                new_report += f"Summary: {result['summary']}\n\n"
            
            new_report += "## Synthesis of Findings\n\n"
            new_report += "### Main Themes\n"
            new_report += "* Summary points from across sources\n"
            new_report += "* Common findings and patterns\n"
            new_report += "* Key differences between sources\n\n"
            
            current_report = state.get("report", "")
            if current_report:
                new_report = current_report + "\n\n" + new_report

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
            report_content = tool_call["args"].get("report", "")
            
            if state["resources"]:
                report_content += "\n\n## References and Sources\n\n"
                for idx, resource in enumerate(state["resources"], 1):
                    report_content += f"{idx}. [{resource['title']}]({resource['url']})\n"
                    report_content += f"   - Summary: {resource['description']}\n\n"
            return {
                "report": report_content,
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
                state["report"] = ""
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
        
        elif tool_call["name"] == "save_report_pdf":
            result = await pdf_tool._arun(state.get("report", ""))
            return {
                "messages": [response, ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=f"""
===================================
AGENT: PDF Generator
ACTION: Save Report
===================================

{result}"""
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
    
    
    
    