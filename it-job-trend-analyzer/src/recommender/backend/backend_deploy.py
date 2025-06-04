import uvicorn
import logging
import fitz  # pymupdf
import re
import os
from dotenv import load_dotenv, find_dotenv
from langchain_together import ChatTogether
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import psycopg2
from psycopg2 import Error
import pandas as pd
import json
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO

# --- Cấu hình Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Tải Biến môi trường ---
if load_dotenv(find_dotenv('ven.env')):
    logger.info("Successfully loaded environment variables from .env file.")
else:
    logger.warning("Could not find ven.env file. Attempting to use environment variables directly.")

# --- Hằng số API và DB ---
API_KEY = os.getenv("TOGETHER_API_KEY")
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
API_TEMPERATURE = 0.7
API_TIMEOUT_SECONDS = 60
API_MAX_TOKEN = 1024

DB_HOST = os.getenv("PG_HOST", "dpg-d0qo60re5dus739obmv0-a.singapore-postgres.render.com")
DB_NAME = os.getenv("PG_DATABASE", "job_post_database")
DB_USER = os.getenv("PG_USER", "myuser")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("PG_PORT", "5432")

# --- Các hàm xử lý CV (KHÔNG THAY ĐỔI LOGIC) ---
# NOTE: extract_text_from_pdf_fitz cần nhận bytes hoặc file-like object, không phải đường dẫn
def extract_text_from_pdf_fitz(pdf_bytes: bytes):
    """Trích xuất văn bản từ file PDF (dưới dạng bytes)."""
    # fitz.open nhận một file-like object hoặc đường dẫn.
    # BytesIO tạo một file-like object từ bytes.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

def clean_pdf_text(raw_text):
    """Làm sạch văn bản từ PDF."""
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', raw_text)
    email_pattern = r'\b[\w.-]+?@\w+?\.\w+?\b'
    text = re.sub(email_pattern, '', text)
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?){1,3}\d{3,4}'
    text = re.sub(phone_pattern, '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'©.*?topcv\.vn', '', text, flags=re.IGNORECASE)
    return text.strip()

# --- Khởi tạo LLM và Embedding Model (khởi tạo một lần) ---
llm = ChatTogether(
    model=DEFAULT_MODEL_NAME,
    temperature=API_TEMPERATURE,
    max_tokens=API_MAX_TOKEN,
    timeout=API_TIMEOUT_SECONDS,
    max_retries=2,
    api_key=API_KEY
)
logger.info(f"Đã khởi tạo LLM với model: {DEFAULT_MODEL_NAME}")

MODEL_CACHE_DIR = "/app/model_cache"

logger.info("Bắt đầu khởi tạo SentenceTransformer model.")

# Khởi tạo model mà KHÔNG TRUYỀN cache_folder trực tiếp.
# Model sẽ tự động tìm biến môi trường TRANSFORMERS_CACHE (đã đặt trong Dockerfile)
# để quyết định nơi lưu/đọc cache.
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
logger.info("Đã khởi tạo SentenceTransformer model.")


# --- Các Prompts (KHÔNG THAY ĐỔI LOGIC) ---
prompt_system_description = """
You are a professional job description writer.
Given detailed CV information, generate a job description in formal prose that reflects typical job postings.
Focus on summarizing relevant skills, qualifications, education, technologies, tools, and practical experience related to the candidate's expertise.
Do not include personal achievements, leadership titles, awards, competitions, or any personal accolades.
Do not use bullet points or section headers. Write in clear, concise paragraphs.
For certificates, infer and describe the skills and knowledge likely gained rather than simply listing them.
Avoid stating exact numerical values such as GPA, test scores. Instead, describe these qualifications in general terms
Keep the output within 1024 tokens, prioritizing the most relevant and technical information if needed.
Avoid personal narrative style and maintain a neutral, professional tone appropriate for a job description.
"""

prompt_example = """
Example input CV:
PROFILE Linh Trung, Thu Duc, Ho Chi Minh
EDUCATION UNIVERSITY: University of Information Technology (UIT VNU-HCM)
Major: Information Technology
Expected Graduation: 3/2025
GPA: 3.3
TOEIC CERTIFICATE: Listening and Reading: 865
Speaking and Writing: 290
SKILLS PROGRAMMING: Python, C++ , SQL
DATA Spark - Kafka - DBT - Airflow - Docker
ETL - OLAP - OLTP
Data Analysis - Web Scraping - Data Preprocessing - Data Modeling
ADDITIONAL INFORMATION LinkedIn: guyen283/ GitHub: Facebook: oo/ Hackerrank: oo
NGUYEN PHU TRUNG DATA ENGINEER
OBJECTIVE
Having recently completed my studies in IT, I am seeking a Data Engineer position where I can enhance my knowledge and gain practical experience. With a strong passion for data, I also look forward to making meaningful contributions to the company's operations in the long run.
EXPERIENCE
WEB CRAWLER // 9/2024 - Present
Freelancer at Global Tech
Assigned Tasks:
Developed Python scripts to crawl, extract, and transform data from multiple sources based on predefined structures.
Researched and learned new tools and technologies as Apache Nifi, Neo4j, SparQL.
Technologies and skills:
Python: Selenium, Pandas Data Preprocessing, Web Scraping
TRAFFIC VIOLATION DETECTION // 5/2024 - 7/2024
Members: 4 - Leader
My Contributions:
Configured pretrained YOLOv10 for custom training with new labels.
Designed two Kafka topics for storing, sending and displaying frames after prediction.
Configured Spark to retrieve frames from the first topic, detect violations using a user-defined function and toward the results to the next topic.
Built a pipeline for automatically storing detected frames in MongoDB.
Technologies and skills:
Python Kafka, Spark, MongoDB
Source: DAILY AUTOMATED DATA PIPELINE // 11/2024 - Present
Members: 4
My Contributions:
Developed an API-based web crawler using Python-Selenium and FastAPI to collect job data from various sources.
Stored and processed data in ClickHouse, ensuring efficient querying and analysis.
Built a structured data model using DBT, transforming raw data into meaningful insights.
Automated the daily data collection and processing workflow using Apache Airflow and Docker.
Technologies and skills:
Python, SQL
ClickHouse, DBT, Airflow, Docker
ETL, Data Preprocessing
topcv.vn
Expected output:
University graduate, major in Information Technology, good English proficiency, GPA trên 3.0
Proficient in programming languages including Python, C++, and SQL, with practical experience applying them in data engineering tasks.
Skilled in building and managing ETL pipelines, data preprocessing, OLAP and OLTP systems to support data analysis and modeling.
Experienced with big data tools and frameworks such as Apache Spark, Kafka, DBT, ClickHouse, and MongoDB to handle distributed data processing and storage.
Competent in web scraping and data extraction techniques using Python libraries like Selenium and Pandas for automated data collection from various sources.
Developed API-based data crawlers using FastAPI to streamline data ingestion and processing workflows.
Familiar with containerization and workflow orchestration tools including Docker and Apache Airflow, enabling automation of data pipelines and deployment.
Applied practical knowledge by leading and collaborating on projects involving traffic violation detection using YOLOv10 integrated with Kafka and Spark for real-time data streaming and analysis.
Demonstrated capability to design Kafka topics and build pipelines for efficient frame processing and storage in MongoDB databases.
Passionate about continuous learning, improving data engineering skills, and contributing meaningfully to organizational data operations.
Strong analytical thinking, problem-solving skills, and ability to work effectively in team environments under deadlines.
"""

prompt_system_reranking_feature = """
You are a recruiter specializing in reading CVs. I will provide you with a piece of text extracted from a candidate’s CV. From that, I need you to extract the following attributes, and you must strictly follow the exact output requirements stated below.
province: Return the name of the province in Vietnam where the candidate currently lives, in Vietnamese. For example: "Hà Nội", "Hồ Chí Minh", …
language: Return the languages the candidate is proficient in. Extract this as a list of languages based on both their current country of residence and any languages explicitly mentioned in the CV. For example: [“tiếng Việt”, “tiếng Anh”, "tiếng Pháp”, “tiếng Trung”,….]
job_type: Extract the type of job the candidate is looking for, if mentioned. You must return one of the following values:
"Toàn thời gian”, "Thực tập", "Bán thời gian”, "Làm tại nhà", "Thời vụ”. If no suitable value is found, return “”.
academic_level: Extract the candidate’s education level and return one of the following exact values:
"Đại Học trở lên”, "Cao Đẳng trở lên”,
"Trung học phổ thông (Cấp 3) trở lên”,
"Cao học trở lên”, "Trung cấp trở lên”.
For example, if the candidate has a university degree, return "Đại Học trở lên”. If no suitable value is found, return “”.
experience_years: Based on the number of years of working experience, return the value in the format: number + "năm”.
For example, if they have 2 years of experience, return “2 năm”. If they are a recent graduate or have no experience, return "Không yêu cầu”.
role: Based on the role the candidate is applying for in the CV, return that role in English, such as “Data Engineer”, “Data Scientist”, “Software Engineer”,…
tech_stack: Extract the tech stacks the candidate is familiar with and return them as an uppercase list of strings. For example:
["WEB", "HTTP", "CORS", "SELENIUM", "CUCUMBER", "BEHAVE", "CI/CD", "AZURE DEVOPS", "GITHUB ACTIONS", "GITLAB", "PYTEST", "TESTNG", "KUBERNETES", "LOCUST", "METASPLOIT", "LINUX", "AWS", "AZURE", "SQL", "GRAFANA", "OPENTELEMETRY", "PROMETHEUS", "SPLUNK", "JAEGER", "ZIPKIN", "ELASTIC", "AWS X-RAY", "PYTHON", "JS", "GO", "SQL"]
You must return the output strictly in **valid JSON format only**, with the keys: "province", "language", "job_type", "academic_level", "experience_years", "role", and "tech_stack". Do not include any extra explanations or descriptions.
"""

# --- Các hàm tính toán độ tương đồng (KHÔNG THAY ĐỔI LOGIC) ---
def compute_similarity_score_list_and_list(list_1: list, list_2: list) -> float:
    """Tính toán độ tương đồng giữa hai danh sách (ví dụ: tech, language)."""
    set_1 = set(list_1)
    set_2 = set(list_2)
    common_elements = set_1.intersection(set_2)
    if len(set_2) == 0: return 0.0
    return len(common_elements) / len(set_2)

def compute_similarity_score_str_and_str(str_1: str, str_2: str) -> float:
    """Tính toán độ tương đồng giữa hai chuỗi (ví dụ: job_type)."""
    if str_1 == str_2: return 1.0
    else: return 0.0

def compute_similarity_score_str_and_list(str_1: str, list_2: list) -> float:
    """Tính toán độ tương đồng giữa một chuỗi và một danh sách (ví dụ: province)."""
    if str_1 in list_2:
        return 1.0
    else:
        return 0.0

def compute_similarity_score_int_and_int(int_1: int, int_2: int) -> float:
    """Tính toán độ tương đồng giữa hai số nguyên (ví dụ: experience_years, academic_level)."""
    if int_1 >= int_2: return 1.0
    else: return 0.0 

def compute_similarity_score_between_two_vector_embedding(vector_embedding_1: np.ndarray, vector_embedding_2: np.ndarray) -> float:
    """Tính toán độ tương đồng cosine giữa hai vector embedding."""
    if vector_embedding_1.ndim == 1:
        vector_embedding_1 = np.expand_dims(vector_embedding_1, axis=0)
    vector_embedding_1 = vector_embedding_1.astype(np.float32)

    if vector_embedding_2.ndim == 1:
        vector_embedding_2 = np.expand_dims(vector_embedding_2, axis=0)
    vector_embedding_2 = vector_embedding_2.astype(np.float32)
    
    vector_embedding_1_tensor = torch.from_numpy(vector_embedding_1)
    vector_embedding_2_tensor = torch.from_numpy(vector_embedding_2)

    similarities_matrix = util.cos_sim(vector_embedding_1_tensor, vector_embedding_2_tensor)
    similarity_score = similarities_matrix.item()
    
    if similarity_score < 0:
        return 0.0
    else:
        return float(similarity_score)

def is_value_null_like_safe(val):
    """Kiểm tra xem giá trị có giống null hay không (cho mục đích tính điểm)."""
    if isinstance(val, np.ndarray):
        return val.size == 0
    if isinstance(val, list):
        return len(val) == 0 or (len(val) == 1 and isinstance(val[0], str) and val[0] == '')
    if pd.isna(val):
        return True
    if isinstance(val, str):
        return val in ['Null', 'null', '']
    return False

# --- Hàm chính tính toán điểm số (KHÔNG THAY ĐỔI LOGIC) ---
def calculate_reranking_scores(df_rerank: pd.DataFrame, feature_rerank_dict: dict, 
                               cv_description_embedding: np.ndarray, cv_role_embedding: np.ndarray) -> pd.DataFrame:
    """Tính toán điểm độ tương đồng cho từng công việc dựa trên các thuộc tính."""
    df_results = pd.DataFrame(columns=['id', 'similarity_score'])
    df_results['id'] = df_results['id'].astype('Int64')
    df_results['similarity_score'] = df_results['similarity_score'].astype(float)

    # Các ánh xạ đã định nghĩa
    experience_mapping = {
        '1 năm': 1, '3 năm': 3, '4 năm': 4, '2 năm': 2, '5 năm': 5,
        'Trên 5 năm': 5, 'Dưới 1 năm': 0, 'Không yêu cầu': 0
    }
    academic_level_mapping = {
        '': 0, 'Trung cấp trở lên': 1, 'Trung học phổ thông (Cấp 3) trở lên': 2,
        'Cao Đẳng trở lên': 3, 'Đại Học trở lên': 4, 'Cao học trở lên': 5
    }

    # Áp dụng các ánh xạ cho DataFrame job_post nếu chưa được xử lý
    df_rerank['experience_years_mapped'] = [experience_mapping.get(str(x), 0) for x in df_rerank['experience_years']]
    df_rerank['academic_level_mapped'] = [academic_level_mapping.get(str(x), 0) for x in df_rerank['academic_level']]

    # Chuẩn hoá thuộc tính experience_years và academic_level trong CV
    cv_experience_years_mapped = experience_mapping.get(feature_rerank_dict['experience_years'], 0)
    cv_academic_level_mapped = academic_level_mapping.get(feature_rerank_dict['academic_level'], 0)

    for i in range(len(df_rerank)):
        current_id = df_rerank['id'].iloc[i]

        province_score = 1.0
        current_province = df_rerank['province'].iloc[i]
        if not is_value_null_like_safe(current_province):
            province_score = compute_similarity_score_str_and_list(feature_rerank_dict['province'], current_province)

        language_score = 1.0
        current_language = df_rerank['language'].iloc[i]
        if not is_value_null_like_safe(current_language):
            language_score = compute_similarity_score_list_and_list(feature_rerank_dict['language'], current_language)

        job_type_score = 1.0
        current_job_type = df_rerank['job_type'].iloc[i]
        if not is_value_null_like_safe(current_job_type):
            job_type_score = compute_similarity_score_str_and_str(feature_rerank_dict['job_type'], current_job_type)

        academic_level_score = 1.0
        current_academic_level = df_rerank['academic_level_mapped'].iloc[i]
        if not is_value_null_like_safe(current_academic_level):
            academic_level_score = compute_similarity_score_int_and_int(cv_academic_level_mapped, current_academic_level)

        experience_years_score = 1.0
        current_experience_years = df_rerank['experience_years_mapped'].iloc[i]
        if not is_value_null_like_safe(current_experience_years):
            experience_years_score = compute_similarity_score_int_and_int(cv_experience_years_mapped, current_experience_years)

        # Cột 'tech' trong câu truy vấn SQL đã bị xóa để tránh lỗi,
        # Nếu cột này không tồn tại trong DB, 'current_tech' sẽ là NaN hoặc không có.
        # Logic tính toán 'tech_score' sẽ phụ thuộc vào việc bạn có lấy được cột này từ DB hay không.
        # Ở đây tôi GIỮ NGUYÊN LOGIC CỦA BẠN, nếu 'current_tech' là null-like, score sẽ là 1.0.
        tech_score = 1.0
        # Đảm bảo cột 'tech' tồn tại trong df_rerank trước khi truy cập
        if 'tech' in df_rerank.columns:
            current_tech = df_rerank['tech'].iloc[i]
            if not is_value_null_like_safe(current_tech):
                tech_score = compute_similarity_score_list_and_list(feature_rerank_dict['tech_stack'], current_tech)
        else:
            logger.warning("Cảnh báo: Cột 'tech' không tồn tại trong DataFrame df_rerank. Tech score sẽ được tính là 1.0.")


        role_score = 1.0
        current_role_embedding = df_rerank['role_en_embedding'].iloc[i]
        if not is_value_null_like_safe(current_role_embedding):
            role_en_embedding_job = np.array(current_role_embedding).astype(np.float32)
            role_score = compute_similarity_score_between_two_vector_embedding(cv_role_embedding.astype(np.float32), role_en_embedding_job)

        description_score = 1.0
        current_description_embedding = df_rerank['description_en_embedding'].iloc[i]
        if not is_value_null_like_safe(current_description_embedding):
            description_en_embedding_job = np.array(current_description_embedding).astype(np.float32)
            description_score = compute_similarity_score_between_two_vector_embedding(cv_description_embedding.astype(np.float32), description_en_embedding_job)

        # Trọng số (KHÔNG THAY ĐỔI)
        total_score = (30 * description_score) + \
                      (20 * tech_score) + \
                      (20 * role_score) + \
                      (10 * experience_years_score) + \
                      (5 * academic_level_score) + \
                      (5 * province_score) + \
                      (5 * language_score) + \
                      (5 * job_type_score)

        df_results.loc[len(df_results)] = [current_id, float(total_score)]
    
    df_results = df_results.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    return df_results

# --- Hàm chính thực hiện toàn bộ quy trình ---
# Hàm này sẽ được thay đổi để nhận dữ liệu CV dưới dạng bytes
def recommend_jobs_from_cv(cv_file_bytes: bytes):
    """
    Thực hiện toàn bộ quy trình từ trích xuất CV đến đề xuất công việc.
    Args:
        cv_file_bytes (bytes): Nội dung file PDF của CV dưới dạng bytes.
    Returns:
        str: Chuỗi JSON chứa danh sách các công việc được đề xuất.
    """
    conn = None
    cur = None
    try:
        logger.info(f"Bắt đầu quy trình đề xuất công việc cho CV.")

        # --- Trích xuất và làm sạch văn bản từ CV ---
        extracted_text = extract_text_from_pdf_fitz(cv_file_bytes)
        cleaned_text = clean_pdf_text(extracted_text)
        logger.info("Đã trích xuất và làm sạch văn bản CV.")

        # --- Gọi API để lấy CV Description và Re-ranking Features ---
        messages_description = [
            ("system", prompt_system_description),
            ("human", prompt_example + "\n\nNow analyze this CV:\n" + cleaned_text)
        ]
        cv_description = llm.invoke(messages_description).content
        logger.info("Đã tạo CV Description từ LLM.")
        
        messages_rerank = [
            ("system", prompt_system_reranking_feature),
            ("human", "\n\nHere is the extracted CV text:\n" + cleaned_text)
        ]
        feature_rerank_json_raw = llm.invoke(messages_rerank).content
        logger.info("Đã trích xuất Re-ranking Features từ LLM (dạng JSON string).")

        # Làm sạch chuỗi JSON và chuyển đổi thành dict
        start_index = feature_rerank_json_raw.find('{')
        end_index = feature_rerank_json_raw.rfind('}')
        json_content_cleaned = ""
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_content_cleaned = feature_rerank_json_raw[start_index : end_index + 1].strip()
        else:
            logger.error(f"Cảnh báo: Không tìm thấy cấu trúc JSON hợp lệ trong chuỗi từ LLM: {feature_rerank_json_raw}")
            return json.dumps({"error": "Không thể phân tích JSON từ LLM"}, indent=4, ensure_ascii=False)

        feature_rerank_dict = json.loads(json_content_cleaned)
        logger.info("Đã phân tích Re-ranking Features thành Python dictionary.")

        # --- Tạo Embeddings ---
        cv_description_embedding = embedding_model.encode(cv_description, show_progress_bar=False)
        cv_role_embedding = embedding_model.encode(feature_rerank_dict['role'], show_progress_bar=False)
        logger.info("Đã tạo embeddings cho CV Description và Role.")

        # --- Retrieval bằng ChromaDB (KHÔNG THAY ĐỔI N_RESULTS) ---
        RENDER_HOST = "description-job-chroma-server-latest.onrender.com"
        RENDER_PORT = 443
        COLLECTION_NAME = "job_descriptions_vector_embedding"
        
        client = chromadb.HttpClient(host=RENDER_HOST, port=RENDER_PORT, ssl=True)
        logger.info(f"Đã kết nối tới ChromaDB server tại: {RENDER_HOST}:{RENDER_PORT}")

        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            logger.info(f"Đã lấy thành công Collection '{collection.name}'. Số lượng bản ghi: {collection.count()}")
        except Exception as e:
            logger.error(f"Lỗi khi lấy ChromaDB collection: {e}")
            return json.dumps({"error": "Không thể kết nối hoặc lấy collection từ ChromaDB"}, indent=4, ensure_ascii=False)
        # Truy vấn top 10
        results = collection.query(
            query_embeddings=[cv_description_embedding.tolist()],
            n_results=10, 
            include=['metadatas']
        )
        list_id_job_str = results['ids'][0]
        list_id_job_as_int = [int(id_str) for id_str in list_id_job_str]
        logger.info(f"Đã truy xuất {len(list_id_job_as_int)} ID công việc từ ChromaDB.")
        
        if not list_id_job_as_int:
            logger.warning("Không tìm thấy ID công việc nào từ ChromaDB. Trả về kết quả rỗng.")
            return json.dumps([], indent=4, ensure_ascii=False)


        # --- Kết nối PostgreSQL và Truy vấn dữ liệu cho Re-ranking ---
        logger.info(f"Đang cố gắng kết nối tới PostgreSQL tại {DB_HOST}:{DB_PORT}/{DB_NAME}...")
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            sslmode='require'
        )
        cur = conn.cursor()
        logger.info("Đã kết nối thành công tới PostgreSQL!")

        feature_table_name = 'table_feature_checking'
        job_post_table_name = 'job_post'
        id_column_name = 'id'
        job_post_columns_to_select = "jp.tech"
        
        sql_query_rerank = f"""
        WITH RankedFeatures AS (
            SELECT {id_column_name}, province, language, job_type, academic_level, experience_years, role_en_embedding, description_en_embedding
            FROM {feature_table_name}
            WHERE {id_column_name} = ANY(%s)
            ORDER BY ARRAY_POSITION(%s, {id_column_name})
        )
        SELECT
            rf.*, -- Chọn tất cả các cột từ RankedFeatures (bao gồm id, province, language, ...)
            {job_post_columns_to_select} -- Các cột bổ sung từ bảng job_post
        FROM
            RankedFeatures rf
        LEFT JOIN
            {job_post_table_name} jp ON rf.{id_column_name} = jp.{id_column_name};
        """
        
        cur.execute(sql_query_rerank, (list_id_job_as_int, list_id_job_as_int))
        selected_columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        df_rerank = pd.DataFrame(rows, columns=selected_columns)
        logger.info(f"Đã truy vấn {len(df_rerank)} bản ghi từ DB để re-ranking.")
        
        if df_rerank.empty:
            logger.warning("Không có dữ liệu công việc nào được truy vấn từ PostgreSQL cho re-ranking. Trả về kết quả rỗng.")
            return json.dumps([], indent=4, ensure_ascii=False)

        # --- Tính toán điểm Re-ranking ---
        final_df_results = calculate_reranking_scores(df_rerank, feature_rerank_dict, 
                                                     cv_description_embedding, cv_role_embedding)
        logger.info("Đã hoàn tất tính toán điểm re-ranking.")

        # --- Ghép nối lại với thông tin chi tiết của job_post và xuất JSON ---
        top_job_ids = final_df_results['id'].astype(int).tolist() # Đảm bảo là list of int
        


        sql_query_final_jobs = f"SELECT * FROM job_post WHERE id = ANY(%s) ORDER BY ARRAY_POSITION(%s, id);"
        cur.execute(sql_query_final_jobs, (top_job_ids, top_job_ids))
        final_job_columns = [desc[0] for desc in cur.description]
        final_job_rows = cur.fetchall()
        df_final_jobs_from_db = pd.DataFrame(final_job_rows, columns=final_job_columns)
        # đảm bảo id là int
        if id_column_name in df_final_jobs_from_db.columns:
            df_final_jobs_from_db[id_column_name] = df_final_jobs_from_db[id_column_name].astype('Int64')
        logger.info(f"Đã truy vấn {len(df_final_jobs_from_db)} bản ghi chi tiết công việc cuối cùng.")

        # Ghép nối similarity_score với thông tin job_post đầy đủ
        final_recommended_df = pd.merge(final_df_results, df_final_jobs_from_db, on='id', how='left')
        final_recommended_df = final_recommended_df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
        final_recommended_df['id'] = final_recommended_df['id'].astype(int)
        # Chuyển DataFrame thành định dạng JSON
        json_output = final_recommended_df.to_json(orient='records', indent=4, force_ascii=False)
        logger.info("Đã tạo JSON kết quả đề xuất công việc.")
        
        return json_output

    except (Exception, Error) as error:
        logger.error(f"Lỗi trong quá trình đề xuất công việc: {error}", exc_info=True)
        # Trả về JSON lỗi cho frontend
        return json.dumps({"error": str(error)}, indent=4, ensure_ascii=False)
    finally:
        if cur:
            cur.close()
            logger.info("Cursor PostgreSQL đã đóng.")
        if conn:
            conn.close()
            logger.info("Kết nối PostgreSQL đã đóng.")

# --- Khởi tạo FastAPI app ---
app = FastAPI(
    title="CV Job Recommendation API",
    description="API để đề xuất công việc dựa trên phân tích CV.",
    version="1.0.0",
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is healthy and running!"}
# Endpoint để kiểm tra trạng thái API

@app.get("/")
async def root():
    return {"message": "Job Recommendation API is running!"}

# Endpoint nhận file PDF từ frontend
@app.post("/recommend_jobs/")
async def upload_cv_and_recommend_jobs(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF.")
    
    try:
        # Đọc nội dung file PDF dưới dạng bytes
        pdf_bytes = await file.read()
        
        # Gọi hàm chính để xử lý và đề xuất công việc
        recommendations_json = recommend_jobs_from_cv(pdf_bytes)
        
        # Parse JSON string to Python dict to return proper JSON response
        return json.loads(recommendations_json)

    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý request: {e}", exc_info=True)
        # Nếu có lỗi, trả về HTTP 500 với thông báo lỗi
        raise HTTPException(status_code=500, detail=str(e))

# --- Khối thực thi chính khi chạy script ---
if __name__ == "__main__":
    # Để chạy ứng dụng FastAPI, bạn dùng lệnh sau trong terminal:
    # uvicorn your_script_name:app --host 0.0.0.0 --port 8000 --reload
    # (uvicorn backend_deploy:app --host 0.0.0.0 --port 8000 --reload)
    # (thay your_script_name bằng tên file .py này, ví dụ: main.py)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)