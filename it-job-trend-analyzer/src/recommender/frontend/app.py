import streamlit as st
import requests
import json
import io
from datetime import datetime

# --- Cấu hình trang Streamlit ---
st.set_page_config(
    page_title="Công Cụ Đề Xuất Việc Làm CV",
    page_icon="📄",
    layout="wide", # Sử dụng layout rộng hơn
    initial_sidebar_state="collapsed"
)

# --- Thêm Custom CSS để chỉnh màu sắc, font, và hiệu ứng ---
st.markdown("""
<style>
    /* Tổng thể ứng dụng */
    .stApp {
        background-color: #F8F4E3; /* Màu vàng be làm nền chính */
        color: #1A1A1A; /* Màu chữ đen đậm */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Tiêu đề chính */
    h1 {
        color: #1A1A1A;
        font-size: 2.8em;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2, h3, h4, h5, h6 {
        color: #1A1A1A;
        font-weight: 700;
    }
    p {
        color: #333333;
        line-height: 1.8;
        font-size: 1.1em;
    }

    /* Nút chính (Tải file, Tìm kiếm) */
    .stButton > button {
        background-color: #FFD700; /* Vàng tươi cho nút */
        color: #333333; /* Chữ đen */
        border-radius: 10px;
        font-weight: bold;
        padding: 12px 30px;
        border: 2px solid #CCA300; /* Viền vàng đậm */
        box-shadow: 0 4px 10px rgba(255, 215, 0, 0.4);
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FFC400; /* Vàng đậm hơn khi hover */
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(255, 215, 0, 0.6);
    }

    /* File uploader button */
    .stFileUploader > div > button {
        background-color: #32CD32; /* Xanh lá cây tươi */
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        border: 2px solid #228B22; /* Viền xanh lá đậm */
        box-shadow: 0 4px 10px rgba(50, 205, 50, 0.4);
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
    }
    .stFileUploader > div > button:hover {
        background-color: #2E8B57; /* Xanh đậm hơn khi hover */
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(50, 205, 50, 0.6);
    }

    /* Điều chỉnh các expander (job cards) */
    .stExpander {
        margin-bottom: 20px; /* Khoảng cách giữa các expander */
        border-radius: 12px;
        overflow: hidden; /* Đảm bảo bo góc đẹp */
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); /* Đổ bóng nhẹ cho cả card */
        border: 2px solid #FFD700; /* Thêm viền vàng cho mỗi expander */
    }

    /* Header của expander (phần có thể click để mở/đóng) */
    .stExpander > div[data-testid="stExpanderToggle"] {
        color: #1A1A1A !important;
        font-size: 1.1em; /* Tăng kích thước chữ trong tiêu đề expander */
        font-weight: bold;
        border-radius: 10px 10px 0 0; /* Bo góc trên, không bo góc dưới */
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: background-color 0.3s ease, box-shadow 0.2s ease, transform 0.2s ease;
    }
    /* Hover state cho header expander */
    .stExpander > div[data-testid="stExpanderToggle"]:hover {
        background-color: #D0D0D0; /* Xám nhạt hơn khi hover */
        box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
        transform: translateY(-3px);
    }

    /* Hiệu ứng xen kẽ màu cho header expander */
    div.stExpander:nth-of-type(odd) > div[data-testid="stExpanderToggle"] {
        background-color: #F8F8F8; /* Màu nền nhạt hơn cho hàng lẻ */
    }
    div.stExpander:nth-of-type(even) > div[data-testid="stExpanderToggle"] {
        background-color: #EFEFEF; /* Màu nền đậm hơn cho hàng chẵn */
    }

    /* CSS cho Job Title LỚN VÀ CÓ MÀU BÊN TRONG NỘI DUNG EXPANDER */
    .job-title-content {
        color: #007BFF; /* Xanh nước biển tươi */
        font-size: 1.8em; /* To hơn để dễ đọc */
        font-weight: 800; /* Rất đậm */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 20px; /* Khoảng cách với các thông tin khác */
        border-bottom: 2px solid #ADD8E6; /* Kẻ dưới job title */
        padding-bottom: 10px;
    }

    /* Nội dung của expander - Tạo hiệu ứng kẻ bảng nhẹ */
    .stExpander > div[data-testid="stExpanderContent"] {
        background-color: #FFFFFF; /* Nền trắng cho nội dung */
        border: 1px solid #E0E0E0; /* Giữ lại border nhẹ cho nội dung */
        border-top: none; /* Bỏ border trên vì đã có viền của expander */
        border-radius: 0 0 12px 12px; /* Bo góc dưới */
        padding: 1.5rem;
        margin-top: 0; /* Đảm bảo không có khoảng cách âm */
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }
    .stExpander > div[data-testid="stExpanderContent"] p {
        margin-bottom: 8px; /* Khoảng cách giữa các dòng thông tin */
        font-size: 1.05em;
        color: #4A4A4A;
        padding: 5px 0; /* Padding nhẹ cho mỗi dòng */
        border-bottom: 1px dashed #F0F0F0; /* Đường kẻ mờ giữa các dòng */
    }
    .stExpander > div[data-testid="stExpanderContent"] p:last-of-type {
        border-bottom: none; /* Bỏ kẻ ở dòng cuối cùng */
    }
    .stExpander > div[data-testid="stExpanderContent"] strong {
        color: #1A1A1A;
        min-width: 120px; /* Đảm bảo các nhãn thẳng hàng */
        display: inline-block; /* Để min-width có tác dụng */
    }
    .stExpander > div[data-testid="stExpanderContent"] a {
        color: #007BFF;
        text-decoration: none;
        font-weight: bold;
        transition: color 0.2s ease;
    }
    .stExpander > div[data-testid="stExpanderContent"] a:hover {
        color: #0056b3;
        text-decoration: underline;
    }

    /* Màu cho điểm phù hợp (score) */
    .similarity-score-display {
        color: #28A745; /* Màu xanh lá cây đậm */
        font-weight: bold;
        font-size: 1.1em;
        padding-left: 5px;
    }
    /* Điều chỉnh p chứa điểm phù hợp để nó nổi bật hơn */
    .stExpander > div[data-testid="stExpanderContent"] p.score-line {
        background-color: #E6FFE6; /* Nền xanh lá nhạt cho dòng điểm */
        border-radius: 5px;
        padding: 10px 15px;
        margin-top: 15px;
        border: 1px solid #C8E6C9;
        font-weight: bold;
    }


    /* Custom CSS cho thanh cuộn (scrollbar) */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

</style>
""", unsafe_allow_html=True)


# --- URL của API Hugging Face Space của bạn ---
API_BASE_URL = 'https://unglong-api-recommender-job.hf.space'
API_RECOMMEND_ENDPOINT = f"{API_BASE_URL}/recommend_jobs/"

# --- Lấy Access Token từ Streamlit Secrets ---
try:
    HF_ACCESS_TOKEN = st.secrets["HF_ACCESS_TOKEN"]
    headers = {"Authorization": f"Bearer {HF_ACCESS_TOKEN}"}
except KeyError:
    st.error("Lỗi: Không tìm thấy Hugging Face Access Token. Vui lòng thêm 'HF_ACCESS_TOKEN' vào file .streamlit/secrets.toml hoặc vào Space secrets.")
    st.stop()

# --- Ảnh đại diện ---
CAT_IMAGE_URL = "https://i.pinimg.com/736x/a5/84/1b/a5841b2cf13fd7592fff39b912204fd0.jpg"

# --- Tiêu đề và giới thiệu ---
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(CAT_IMAGE_URL, width=100)
with col2:
    st.title("Công Cụ Đề Xuất Việc Làm từ CV của Bạn")

st.markdown("""
Chào mừng bạn đến với công cụ đề xuất việc làm thông minh! Vui lòng tải lên CV của bạn (định dạng PDF) để chúng tôi phân tích và đề xuất các vị trí công việc phù hợp nhất.
""")

# --- Khu vực tải file ---
uploaded_file = st.file_uploader(
    "Chọn CV của bạn (định dạng PDF)",
    type="pdf",
    help="Chỉ chấp nhận file PDF. Kích thước tối đa được khuyến nghị: 5MB."
)

# --- Nút tìm kiếm ---
if st.button("🔍 Tìm kiếm Việc làm", use_container_width=True):
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            with st.spinner("Đang phân tích CV và tìm kiếm việc làm... Vui lòng chờ trong giây lát."):
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}

                try:
                    response = requests.post(API_RECOMMEND_ENDPOINT, files=files, headers=headers, timeout=120)

                    if response.status_code == 200:
                        data = response.json()

                        if data and "error" in data:
                            st.error(f"❌ Rất tiếc, đã có lỗi xảy ra trong quá trình xử lý CV của bạn: {data['error']}. "
                                     "Chúng tôi đang cố gắng khắc phục sự cố này. Mong bạn thông cảm và thử lại sau ít phút nhé.")
                        elif not isinstance(data, list) or len(data) == 0:
                            st.warning("Không tìm thấy công việc phù hợp nào cho CV của bạn. "
                                       "Bạn có thể thử tải lên một CV khác hoặc điều chỉnh nội dung CV để có kết quả tốt hơn nhé.")
                        else:
                            st.success("🎉 Đã tìm thấy các công việc phù hợp! Cảm ơn bạn đã sử dụng dịch vụ của chúng tôi.")

                            # --- Hiển thị Top 10 công việc (không còn khung trắng bao ngoài) ---
                            st.subheader("Top 10 công việc đề xuất:")

                            # Sử dụng st.container để có thanh cuộn nếu nội dung dài
                            with st.container(height=700): # Bạn có thể điều chỉnh chiều cao
                                for i, job in enumerate(data[:10]):
                                    job_title_display = job.get('job_title', 'Không có tiêu đề')
                                    similarity_score_display = job.get('similarity_score', 0.0)
                                    
                                    # Chuyển đổi timestamp deadline
                                    deadline_timestamp = job.get('deadline')
                                    deadline_display = "N/A"
                                    if deadline_timestamp:
                                        try:
                                            # Chuyển đổi từ mili giây sang giây rồi sang datetime
                                            dt_object = datetime.fromtimestamp(deadline_timestamp / 1000)
                                            deadline_display = dt_object.strftime("%d/%m/%Y")
                                        except Exception:
                                            deadline_display = "Không xác định"

                                    # Chuyển đổi timestamp date_crawl_module_1
                                    date_crawl_timestamp = job.get('date_crawl_module_1')
                                    date_crawl_display = "N/A"
                                    if date_crawl_timestamp:
                                        try:
                                            dt_object = datetime.fromtimestamp(date_crawl_timestamp / 1000)
                                            date_crawl_display = dt_object.strftime("%d/%m/%Y")
                                        except Exception:
                                            date_crawl_display = "Không xác định"


                                    # Tiêu đề expander
                                    expander_title_text = f"**{i+1}. {job_title_display}**"

                                    # st.expander không dùng unsafe_allow_html=True ở tiêu đề chính
                                    with st.expander(expander_title_text, expanded=False):
                                        # Hiển thị Job Title LỚN VÀ CÓ MÀU ngay đầu nội dung expander
                                        st.markdown(f"<h3 class='job-title-content'>{job_title_display}</h3>", unsafe_allow_html=True)

                                        # Hiển thị điểm phù hợp với màu sắc
                                        st.markdown(f"<p class='score-line'><strong>Điểm phù hợp:</strong> <span class='similarity-score-display'>{similarity_score_display:.2f}</span></p>", unsafe_allow_html=True)

                                        # Các thông tin khác, tạo cảm giác "kẻ bảng" bằng CSS
                                        st.markdown(f"<p><strong>Công ty:</strong> {job.get('company_name', 'N/A')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Địa điểm:</strong> {', '.join(job.get('province', ['N/A'])) if job.get('province') else 'N/A'}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Mức lương:</strong> {job.get('salary', 'Thương lượng')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Loại công việc:</strong> {job.get('job_type', 'N/A')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Cấp bậc học vấn:</strong> {job.get('academic_level', 'N/A')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Kinh nghiệm:</strong> {job.get('experience_years', 'Không yêu cầu')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Cấp bậc:</strong> {job.get('level', 'N/A')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Ngôn ngữ:</strong> {', '.join(job.get('language', ['N/A'])) if job.get('language') else 'N/A'}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Tech Stack:</strong> {', '.join(job.get('tech', ['N/A'])) if job.get('tech') else 'N/A'}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Vai trò:</strong> {job.get('role', 'N/A')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Tổng số giờ làm việc:</strong> {job.get('total_work_hour', 'N/A')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Thời gian làm việc trong tuần:</strong> {job.get('time_range', 'N/A')}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Hạn nộp hồ sơ:</strong> {deadline_display}</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p><strong>Số lượng tuyển:</strong> {job.get('recruitment_count', 'N/A')}</p>", unsafe_allow_html=True)

                                        # Phần lợi ích (benefits)
                                        benefits = job.get('benefits', 'Không có thông tin.')
                                        if benefits and benefits != 'null': # Kiểm tra nếu benefits không rỗng và không phải 'null'
                                            st.markdown(f"**Lợi ích:**")
                                            # Chia nhỏ lợi ích thành các gạch đầu dòng nếu có thể
                                            benefits_list = [b.strip() for b in benefits.split('....') if b.strip()]
                                            if benefits_list:
                                                benefits_markdown = "".join([f"- {b}<br>" for b in benefits_list])
                                                st.markdown(f"<p>{benefits_markdown}</p>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"<p>{benefits}</p>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<p><strong>Lợi ích:</strong> Không có thông tin.</p>", unsafe_allow_html=True)


                                        # Phần mô tả (description)
                                        description = job.get('description', 'Không có mô tả chi tiết.')
                                        st.markdown(f"**Mô tả:**")
                                        st.markdown(f"<p>{description.replace('\\n', '<br>')}</p>", unsafe_allow_html=True) # Thay thế \n bằng <br> để xuống dòng

                                        st.markdown(f"---")
                                        st.markdown(f"[Xem chi tiết bài đăng]({job.get('url_job_posting', '#')})", unsafe_allow_html=True)
                                        st.markdown(f"[Xem hồ sơ công ty]({job.get('company_url', '#')})", unsafe_allow_html=True)

                    else:
                        st.error(f"❌ Rất tiếc, đã có lỗi kết nối đến máy chủ: {response.status_code} - {response.reason}. "
                                 "Chúng tôi đang cố gắng khắc phục sự cố này. Mong bạn thông cảm và thử lại sau ít phút nhé.")
                        st.json(response.json())

                except requests.exceptions.Timeout:
                    st.error("⏰ Yêu cầu API đã hết thời gian. Có vẻ máy chủ đang bận hoặc mạng của bạn không ổn định. "
                             "Vui lòng thử lại sau vài giây nhé.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Không thể kết nối đến máy chủ API. Vui lòng kiểm tra kết nối internet của bạn hoặc URL API. "
                             "Chúng tôi xin lỗi vì sự bất tiện này.")
                except Exception as e:
                    st.error(f"⚠️ Đã xảy ra lỗi không mong muốn: {e}. "
                             "Rất xin lỗi vì sự cố này. Nhóm phát triển của chúng tôi đã nhận được thông báo và sẽ xem xét sớm nhất có thể. "
                             "Cảm ơn sự kiên nhẫn của bạn.")
        else:
            st.error("File tải lên không phải là PDF hợp lệ. Vui lòng chọn một file PDF nhé.")
    else:
        st.warning("Vui lòng tải lên một file CV (PDF) trước khi tìm kiếm công việc nhé.")

st.markdown("---")
st.markdown("Được phát triển bởi Ung Hoàng Long")