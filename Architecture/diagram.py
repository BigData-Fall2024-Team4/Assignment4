from diagrams import Diagram, Cluster, Edge
from diagrams.gcp.storage import GCS
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.container import Docker
from diagrams.onprem.client import User
from diagrams.custom import Custom

# Define the diagram
with Diagram("CFA Architecture Diagram", show=False):
    # Data Ingestion Cluster
    with Cluster("Data Ingestion (Docker)"):
        airflow = Airflow("Apache Airflow")
        bucket = GCS("AWS S3 Bucket")
        
        # Connections within Data Ingestion
        airflow >> Edge(label="Store Data") >> bucket
        bucket >> Edge(label="Retrieve Data") >> airflow

    # FastAPI Cluster for Data Processing
    with Cluster("FastAPI (Docker)"):
        fastapi = Custom("FastAPI", "images/fastapi1.png")
        tavily = Custom("WebAgent", "images/tavily.png")
        fetchariv = Custom("PrivAgent", "images/arxiv.png")
        nvidia_nim = Custom("NVIDIA Embeddings", "images/nvidia.png")
        pinecone = Custom("Pinecone", "images/pinecone.png")
        openai = Custom("OpenAI", "images/openai.png")

        # FastAPI internal connections
        fastapi << Edge(label="webagent") >> tavily
        fastapi << Edge(label="Arxiv Agent") >> fetchariv
        fastapi << Edge(dir="both") >> nvidia_nim
        fastapi << Edge(dir="both") >> pinecone
        fastapi << Edge(dir="both") >> openai

    # CopilotKit Cluster for User Interaction
    with Cluster("CopilotKit (Docker)"):
        copilotkit = Custom("CopilotKit", "images/copilot.png")
        
        # Connections between CopilotKit and FastAPI
        copilotkit << Edge(label="Interact with FastAPI") >> fastapi

        # Connections between CopilotKit and FastAPI
        airflow << Edge(label="Document Selection") << fastapi
        airflow >> Edge(label="Vector Store") >> pinecone
        airflow >> Edge(label="Embeddings") >> nvidia_nim

    # User Interface Cluster
    user = User("User")
    user << Edge(label="Access") >> copilotkit

    # Docker boundaries for clarity
    docker_ingestion = Docker("Docker (Ingestion)")
    docker_fastapi = Docker("Docker (FastAPI)")
    docker_copilotkit = Docker("Docker (CopilotKit)")

    # Docker connections
    docker_ingestion >> airflow
    docker_fastapi >> fastapi
    docker_copilotkit >> copilotkit