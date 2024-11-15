from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path
from env_var import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_BUCKET_NAME,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    NVIDIA_API_KEY
)
import boto3
from pinecone import Pinecone, PodSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

default_args = {
    "owner": "tanvi",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "s3_to_pinecone_vector_storage",
    default_args=default_args,
    description="DAG to process .md files from S3 and store vectors in Pinecone",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 11, 15),
    catchup=False,
)

def download_files_from_s3(**kwargs):
    """Task to download .md files from S3 bucket."""
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    local_dir = Path("/tmp/output")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # List objects in the bucket
    response = s3_client.list_objects_v2(Bucket=AWS_BUCKET_NAME)
    files_downloaded = []
    
    for obj in response.get("Contents", []):
        file_name = obj["Key"]
        if file_name.endswith(".md"):
            local_file_path = local_dir / file_name.split("/")[-1]
            s3_client.download_file(AWS_BUCKET_NAME, file_name, str(local_file_path))
            files_downloaded.append(str(local_file_path))
            logger.info(f"Downloaded: {file_name} to {local_file_path}")
    
    if not files_downloaded:
        raise ValueError("No markdown files found in S3 bucket.")
    
    # Return the list of downloaded files
    return files_downloaded

def process_and_store_vectors(**kwargs):
    """Task to process markdown files and store vectors in Pinecone."""
    from openai import OpenAI

    class NVIDIAEmbeddings:
        def __init__(self):
            """Initialize NVIDIA embeddings client."""
            self.embeddings_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=NVIDIA_API_KEY
            )

        def embed_text(self, text, input_type='passage'):
            """Create embedding using NVIDIA API."""
            response = self.embeddings_client.embeddings.create(
                input=[text],
                model="nvidia/nv-embedqa-e5-v5",
                encoding_format="float",
                extra_body={
                    "input_type": input_type,
                    "truncate": "NONE"
                }
            )
            return response.data[0].embedding

    class PineconeVectorStorage:
        def __init__(self):
            """Initialize Pinecone index."""
            self.pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            self.index_name = "docling-vectors"
            self.embeddings = NVIDIAEmbeddings()

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            self.create_index()
            self.index = self.pc.Index(self.index_name)

        def create_index(self):
            """Create Pinecone index if it doesn't exist."""
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,
                    metric='cosine',
                    spec=PodSpec(
                        environment="gcp-starter",
                        pod_type="starter"
                    )
                )

        def process_file(self, file_path):
            """Process a single markdown file and store vectors in Pinecone."""
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = self.text_splitter.split_text(text)
                vectors = []

                for i, chunk in enumerate(chunks[:50]):  # Process up to 50 chunks
                    embedding = self.embeddings.embed_text(chunk)
                    metadata = {"text": chunk[:1000], "chunk_index": i, "source": str(file_path)}
                    vectors.append((f"chunk_{i}", embedding, metadata))

                    # Batch upserts to Pinecone
                    if len(vectors) >= 10:
                        self.index.upsert(vectors=vectors)
                        vectors = []
                        time.sleep(1)

                # Upsert remaining vectors
                if vectors:
                    self.index.upsert(vectors=vectors)

                return True
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                return False

    downloaded_files = kwargs["ti"].xcom_pull(task_ids="download_files_from_s3")
    vector_storage = PineconeVectorStorage()

    processed_count = 0
    failed_count = 0

    for file in downloaded_files:
        if vector_storage.process_file(file):
            processed_count += 1
        else:
            failed_count += 1

    logger.info(f"Processing Summary: {processed_count} files processed successfully, {failed_count} files failed.")
    print(f"Processing Summary: {processed_count} files processed successfully, {failed_count} files failed.")

# Define tasks
download_task = PythonOperator(
    task_id="download_files_from_s3",
    python_callable=download_files_from_s3,
    provide_context=True,
    dag=dag,
)

process_task = PythonOperator(
    task_id="process_and_store_vectors",
    python_callable=process_and_store_vectors,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
download_task >> process_task
