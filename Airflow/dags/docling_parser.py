from env_var import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_BUCKET_NAME
)
import os
from pathlib import Path
import logging
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import tempfile
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse as parse_date
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions
)
from docling_core.types.doc import ImageRefMode, PictureItem
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from airflow.operators.dummy import DummyOperator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S3PDFProcessor:
    def __init__(self):
        # Configure boto3 with retry strategy
        config = Config(
            retries=dict(
                max_attempts=3,
                mode='adaptive'
            ),
            s3={'addressing_style': 'path'},  # Use path-style addressing
            signature_version='s3v4'  # Use v4 signing
        )
        
        # Initialize time synchronization
        self.time_offset = self._get_ntp_time_offset()
        
        # Initialize S3 client with config
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            config=config
        )
        
        self._register_time_adjustment_handler()
        self.bucket_name = AWS_BUCKET_NAME

    def _get_ntp_time_offset(self):
        """Get time offset using public NTP servers instead of S3"""
        try:
            # Create session with retries
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=0.5)
            session.mount('http://', HTTPAdapter(max_retries=retries))
            
            # Use multiple NTP servers for redundancy
            ntp_servers = [
                'http://worldtimeapi.org/api/timezone/UTC',
                'http://worldtimeapi.org/api/timezone/Etc/UTC'
            ]
            
            for server in ntp_servers:
                try:
                    response = session.get(server, timeout=5)
                    if response.status_code == 200:
                        server_time = datetime.fromisoformat(response.json()['datetime'].replace('Z', '+00:00'))
                        local_time = datetime.now(timezone.utc)
                        offset = (server_time - local_time).total_seconds()
                        logger.info(f"Time offset from NTP: {offset} seconds")
                        return offset
                except Exception as e:
                    logger.warning(f"Failed to get time from {server}: {str(e)}")
                    continue
            
            # If all servers fail, try AWS S3 time as fallback
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    MaxKeys=1
                )
                aws_time = response['ResponseMetadata']['HTTPHeaders']['date']
                aws_time = datetime.strptime(aws_time, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=timezone.utc)
                local_time = datetime.now(timezone.utc)
                offset = (aws_time - local_time).total_seconds()
                logger.info(f"Time offset from AWS fallback: {offset} seconds")
                return offset
            except Exception as e:
                logger.warning(f"Failed to get time from AWS fallback: {str(e)}")
                return 0
                
        except Exception as e:
            logger.warning(f"Failed to calculate time offset: {str(e)}")
            return 0

    def _register_time_adjustment_handler(self):
        """Register event handler to adjust request times"""
        def adjust_request_time(request, **kwargs):
            try:
                # Add the calculated offset to the current time
                adjusted_time = datetime.utcnow().timestamp() + self.time_offset
                amz_date = datetime.fromtimestamp(adjusted_time, tz=timezone.utc)
                
                # Update request headers with adjusted time
                request.headers['X-Amz-Date'] = amz_date.strftime('%Y%m%dT%H%M%SZ')
                
                # Also update the Authorization header's date
                auth_header = request.headers.get('Authorization', '')
                if auth_header and 'Credential=' in auth_header:
                    new_date = amz_date.strftime('%Y%m%d')
                    parts = auth_header.split('Credential=')
                    if len(parts) > 1:
                        cred_parts = parts[1].split('/')
                        if len(cred_parts) > 1:
                            cred_parts[1] = new_date
                            parts[1] = '/'.join(cred_parts)
                            request.headers['Authorization'] = 'Credential='.join(parts)
            except Exception as e:
                logger.warning(f"Failed to adjust request time: {str(e)}")
        
        # Register the event handler
        self.s3_client.meta.events.register('before-sign.s3.*', adjust_request_time)

    def _handle_request_with_retry(self, operation_func, *args, **kwargs):
        """Handle S3 requests with retry logic"""
        max_retries = 3
        retry_delay = 1  # Initial delay in seconds
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                last_error = e
                
                if error_code == 'RequestTimeTooSkewed' and attempt < max_retries - 1:
                    # Recalculate time offset and wait before retry
                    logger.warning(f"Time skew detected (attempt {attempt + 1}), recalibrating...")
                    self.time_offset = self._get_ntp_time_offset()
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    break  # Exit loop for non-time-skew errors or last attempt
        
        # If we got here, all retries failed
        raise last_error


    def download_from_s3(self, s3_key, local_path):
        """Download file from S3 to local path with retry handling"""
        return self._handle_request_with_retry(
            self._download_file_internal,
            s3_key,
            local_path
        )

    def _download_file_internal(self, s3_key, local_path):
        """Internal method to download file"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded {s3_key} from S3")
        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise

    def upload_to_s3(self, local_path, s3_key):
        """Upload file from local path to S3 with retry handling"""
        return self._handle_request_with_retry(
            self._upload_file_internal,
            local_path,
            s3_key
        )

    def _upload_file_internal(self, local_path, s3_key):
        """Internal method to upload file"""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to S3 as {s3_key}")
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    def list_pdf_files(self):
        """List all PDF files in the S3 bucket with retry handling"""
        return self._handle_request_with_retry(self._list_pdf_files_internal)

    def _list_pdf_files_internal(self):
        """Internal method to list PDF files"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='',
                Delimiter='/'
            )
            
            pdf_files = []
            for obj in response.get('Contents', []):
                if obj['Key'].lower().endswith('.pdf'):
                    pdf_files.append(obj['Key'])
            
            return pdf_files
            
        except ClientError as e:
            logger.error(f"Error listing PDF files: {str(e)}")
            raise

    def clean_text(self, text):
        """Clean text content by removing metadata and special characters"""
        if not text or not isinstance(text, str):
            return ""
            
        if text.startswith(('self_ref=', 'parent=', 'children=', 'label=', 'prov=')):
            return ""
        
        text = text.strip()
        text = text.replace('\n\n\n', '\n\n')
        return text

    def process_pdf_with_ocr(self, s3_input_key):
        """Process PDF from S3 and upload results back to S3"""
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Download PDF from S3
                input_file = temp_dir_path / Path(s3_input_key).name
                self.download_from_s3(s3_input_key, str(input_file))
                
                # Setup pipeline options
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = True
                pipeline_options.generate_page_images = False
                pipeline_options.generate_picture_images = True
                
                # Configure OCR
                ocr_options = EasyOcrOptions(force_full_page_ocr=False)
                pipeline_options.ocr_options = ocr_options
                
                # Initialize converter
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
                
                # Convert document
                result = converter.convert(str(input_file))
                
                # Setup output directories
                output_dir = temp_dir_path / "output"
                images_dir = output_dir / "images"
                output_dir.mkdir(exist_ok=True)
                images_dir.mkdir(exist_ok=True)
                
                # Process content
                processed_content = []
                picture_counter = 0
                doc_name = input_file.stem
                
                for element, *level in result.document.iterate_items():
                    if isinstance(element, PictureItem):
                        picture_counter += 1
                        try:
                            image_filename = f"{doc_name}_image_{picture_counter}.png"
                            image_path = images_dir / image_filename
                            with image_path.open("wb") as fp:
                                element.image.pil_image.save(fp, "PNG")
                            processed_content.append(f"\n![Image {picture_counter}](images/{image_filename})\n")
                            
                            # Upload image to S3
                            s3_image_key = f"output/images/{image_filename}"
                            self.upload_to_s3(str(image_path), s3_image_key)
                            
                        except Exception as e:
                            logger.error(f"Failed to save image {picture_counter}: {str(e)}")
                    else:
                        if hasattr(element, 'text') and element.text:
                            clean_content = self.clean_text(element.text)
                            if clean_content:
                                if hasattr(element, 'label'):
                                    if 'SECTION_HEADER' in str(element.label):
                                        clean_content = f"\n## {clean_content}\n"
                                    elif 'HEADER' in str(element.label):
                                        clean_content = f"\n### {clean_content}\n"
                                    elif 'LIST_ITEM' in str(element.label):
                                        clean_content = f"- {clean_content}"
                                
                                processed_content.append(clean_content)
                
                # Save markdown
                markdown_file = output_dir / f"{doc_name}.md"
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {doc_name}\n\n")
                    content = '\n\n'.join(line for line in processed_content if line.strip())
                    f.write(content)
                
                # Upload markdown to S3
                s3_markdown_key = f"output/{doc_name}.md"
                self.upload_to_s3(str(markdown_file), s3_markdown_key)
                
                logger.info(f"Successfully processed {doc_name}")
                return s3_markdown_key
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.exception("Full error traceback:")
            raise

# Airflow task functions
def validate_s3_connection():
    """Validate S3 connection and check for PDF files"""
    processor = S3PDFProcessor()
    pdf_files = processor.list_pdf_files()
    if not pdf_files:
        raise ValueError("No PDF files found in S3 bucket")
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files

def process_all_pdfs():
    """Process all PDFs in the S3 bucket"""
    processor = S3PDFProcessor()
    pdf_files = processor.list_pdf_files()
    
    results = []
    for pdf_file in pdf_files:
        try:
            output_key = processor.process_pdf_with_ocr(pdf_file)
            results.append({
                'input_file': pdf_file,
                'output_file': output_key,
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'input_file': pdf_file,
                'error': str(e),
                'status': 'failed'
            })
    
    return results

# Define Airflow DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'pdf_processing_pipeline',
    default_args=default_args,
    description='Process PDFs from S3 and store results back in S3',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Create Airflow tasks
validate_task = PythonOperator(
    task_id='validate_s3_connection',
    python_callable=validate_s3_connection,
    dag=dag
)

process_pdfs_task = PythonOperator(
    task_id='process_pdfs',
    python_callable=process_all_pdfs,
    dag=dag
)

completion_task = DummyOperator(
    task_id='completion_check',
    dag=dag
)

# Set task dependencies
validate_task >> process_pdfs_task >> completion_task

# For running the script directly (without Airflow)
if __name__ == "__main__":
    try:
        # Validate connection
        pdf_files = validate_s3_connection()
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process PDFs
        results = process_all_pdfs()
        
        # Print results
        for result in results:
            if result['status'] == 'success':
                logger.info(f"Successfully processed {result['input_file']} -> {result['output_file']}")
            else:
                logger.error(f"Failed to process {result['input_file']}: {result['error']}")
                
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise