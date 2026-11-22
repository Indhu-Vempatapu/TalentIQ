import streamlit as st                            # For Web Interface (Front-End)
from pdfminer.high_level import extract_text      # To Extract Text from Resume PDF
from sentence_transformers import SentenceTransformer      # To generate Embeddings of text
from sklearn.metrics.pairwise import cosine_similarity     # To get Similarity Score of Resume and Job Description
from groq import Groq                             # API to use LLM's
import re                                         # To perform Regular Expression Functions
from dotenv import load_dotenv                    # Loading API Key from .env file
import os


# Load environment variables from .env
load_dotenv()

# Fetch the key from the environment
api_key = os.getenv("GROQ_API_KEY")


#  Session States to store values 
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume=""

if "job_desc" not in st.session_state:
    st.session_state.job_desc=""



st.set_page_config(page_title="TalentIQ", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background-color: #e8f0fe;
        }

        .block-container{
        padding-top: 1.5rem !important;
        }

        [data-testid="stForm"] {
            background-color: #FFFFFF;
            padding: 28px;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        }

        [data-testid="stForm"] [data-testid="stVerticalBlock"] {
            gap: 18px;
        }

        .brand-header {
            margin-bottom: 24px;
        }

        .input-heading {
            color: #172554;
            font-size: 19px;
            font-weight: 650;
            margin-bottom: 8px;
        }

        .brand-name {
            font-size: 42px;
            font-weight: 800;
            color: #1D4ED8;
        }

        .brand-tagline {
            font-size: 15px;
            color: #64748B;
        }

        .hero-section h2 {
            font-size: 26px;
            font-weight: 700;
            color: #172554;
            margin-bottom: 8px;
        }

        .hero-section p {
            font-size: 15px;
            color: #475569;
        }

        .section-title {
            font-size: 28px;
            font-weight: 700;
            margin-top: 20px;
            margin-bottom: 15px;
            color: #172554;
        }

        .report-content h2 {
            font-size: 25px;
            font-weight: 600;
            color: #172554;
        }
        
        .score-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            color: #111111;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
            width: 100%;
            box-sizing: border-box;
            min-height: 160px;
            margin-bottom: 12px;
        }

        [data-testid="column"] {
            min-width: 0;
        }

        [data-testid="stHorizontalBlock"] {
            gap: 16px;
        }

        .heading-text {
            color: #1a1a1a;
            font-size: 18px;
            font-weight: 500;
        }
        
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    [data-testid="stFileUploader"] ul li {
        color: #1a1a1a !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
            <div class="brand-header"> 
                <div class="brand-name">💼 TalentIQ</div> 
                <div class="brand-tagline">AI Powered Resume Intelligence</div>
            </div>
            <div class="hero-section">
                <h2>Make your resume stand out.</h2>
                <p> Analyze your resume against any job description and discover opportunities to improve your career prospects.</p>
            </div>
            """, unsafe_allow_html=True)

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        extracted_text = extract_text(uploaded_file)
        return extracted_text
    except Exception as e:
        st.markdown(
        f'<p style="color: #FF4500; font-weight: bold;">Error extracting text from PDF: {str(e)}</p>',
        unsafe_allow_html=True
        )
        return "Could not extract text from the PDF file."


# Function to calculate similarity 
def calculate_similarity_bert(text1, text2):
    ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')      # Use BERT or SBERT or any model you want
    # Encode the texts directly to embeddings
    embeddings1 = ats_model.encode([text1])
    embeddings2 = ats_model.encode([text2])
    
    # Calculate cosine similarity without adding an extra list layer
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity


def get_report(resume,job_desc):
    client = Groq(api_key=api_key)

    
    prompt = f"""
    # Role:
    You are an AI Resume Analyzer. Evaluate a candidate's resume
    against the provided job description.

    # Instructions:
    - Analyze the resume based only on the information provided.
    - Do not invent skills, qualifications, or work experience.
    - Evaluate required skills, education, experience, and responsibilities.
    - Give each requirement a score from 1 to 5.
    - Use ✅ when the requirement is satisfied.
    - Use ❌ when the requirement is not satisfied.
    - Use ⚠️ when there is insufficient information to decide.
    - Explain every evaluation using evidence from the resume.
    - Distinguish internships from full-time professional experience.

    # Scoring Rubric:
    Use the following criteria consistently when assigning scores:
    - 5/5: Strong direct evidence that the candidate meets the requirement.
    - 4/5: Good evidence, with only minor gaps.
    - 3/5: Partial evidence; some required skills or qualifications are demonstrated.
    - 2/5: Limited evidence; significant gaps exist.
    - 1/5: No relevant evidence found in the resume.

    # Additional Scoring Rules:
    - Base every score strictly on the information in the resume.
    - Do not assume the candidate has a skill just because it is related to another skill.
    - Do not award points for technologies that are not mentioned or demonstrated.
    - Consider projects and internships as relevant evidence, but do not treat them as full-time professional experience.
    - Evaluate each job requirement independently.
    - If information is insufficient, use ⚠️ and explain what evidence is missing.
    - Never inflate scores to make the candidate feel better.

    # Required Report Structure:
    ## 1. Resume vs. Job Description Evaluation

    Present a Markdown table with these columns:

    | S.No | Requirement / Responsibility | Score | Status | Reason |

    |---|---|---|---|---|

    Important table formatting rules:
    - Keep the requirement number in a separate first column.
    - Put the complete requirement in the second column.
    - Put the score in the third column.
    - Put the emoji in the fourth column.
    - Put the explanation in the fifth column.
    - Every row must contain exactly five cells.
    - Separate every cell using a pipe (|) symbol.
    - Never combine the requirement number and description.
    - Do not insert line breaks inside individual table rows.
    - Follow the same structure for every row, including soft skills.

    ## 2. Key Strengths

    - Highlight the candidate's strongest matching skills.
    - Mention relevant projects, education, and experience.
    - Explain how these strengths align with the job.

    ## 3. Areas of Improvement

    - Identify missing or insufficiently demonstrated skills.
    - Highlight experience or qualification gaps.
    - Explain what the candidate should work on.

    ## 4. Suggestions to Improve Your Resume

    - Give specific, practical recommendations.
    - Suggest relevant skills or technologies to learn.
    - Recommend improvements to project descriptions and resume content.
    - Prioritize the most important improvements for this role.

    # Rules for Improvement Suggestions:

    - Carefully review the candidate's existing skills, projects,
    education, and experience before making recommendations.
    - Do not recommend learning or adding a skill that the candidate
    already demonstrates sufficiently in the resume.
    - If a required skill is already present, suggest how to highlight
    it better instead of recommending that the candidate learn it.
    - Focus on genuine skill gaps and missing job requirements.
    - Distinguish between a completely missing skill and a skill
    that is present but needs stronger evidence.
    - Do not suggest adding experience, certifications, or achievements
    that the candidate has not actually completed.
    - Make every recommendation specific and relevant to the job.

    # Important Rules:
    - Keep the report professional, clear, and easy to understand.
    - Use appropriate emojis without overusing them.
    - Do not repeat the same information unnecessarily.
    - Include exactly one score in the format x/5 for each
    evaluation row.
    - Do not include additional scores in the other sections.

    # Inputs:

    Candidate Resume:
    {resume}

    Job Description:
    {job_desc}
    """


    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            temperature=0,
            seed=42
        )

        report = chat_completion.choices[0].message.content

        if not report:
            st.error("The AI returned an empty report. Please try again.")
            return None

        return report

    except Exception as e:
        print(f"Groq API Error: {e}")
        st.error(
            "Unable to generate the analysis report. "
            "Please try again later."
        )
        return None


def extract_scores(text):

    # Find only the Resume vs. Job Description Evaluation section
    section = re.search(
        r'##\s*1\.\s*Resume vs\. Job Description Evaluation(.*?)(?=##\s*2\.|$)',
        text,
        re.DOTALL | re.IGNORECASE
    )

    # Return an empty list if the section is missing
    if not section:
        return []

    evaluation_text = section.group(1)

    # Find scores in the format x/5
    pattern = r'\b(\d+(?:\.\d+)?)\s*/\s*5\b'

    matches = re.findall(pattern, evaluation_text)

    # Convert scores into numbers and keep only valid scores
    scores = [
        float(match)
        for match in matches
        if 1 <= float(match) <= 5
    ]

    return scores


# Displays Form only if the form is not submitted
if not st.session_state.form_submitted:
    with st.form("my_form"):

        # Taking input a Resume (PDF) file 
         st.markdown('<div class="input-heading">📄 Upload Your Resume </div>', unsafe_allow_html=True)
         resume_file = st.file_uploader("Upload your resume", type=["pdf"], label_visibility="collapsed")
         st.caption("For best results, upload a text-based PDF. "
                    "Scanned image-only PDFs may not produce extractable text.")

         st.markdown('<h4 class="input-heading">📝 Job Description </h4>', unsafe_allow_html=True)
         st.session_state.job_desc = st.text_area("Job Description", placeholder="Place the complete job description here...", height=220, label_visibility="collapsed")

         submitted = st.form_submit_button("🚀Analyze Resume", type="primary", width="stretch")

         
         if submitted:

            # Check whether the resume is uploaded
            if resume_file is None:
                st.error("Please upload your resume in PDF format.")

            # Check whether the job description is entered
            elif not st.session_state.job_desc.strip():
                st.error("Please enter the job description.")

            else:
                # Extract text from the uploaded PDF
                st.session_state.resume = extract_pdf_text(resume_file)

                # Check whether text was extracted successfully
                if not st.session_state.resume.strip() or st.session_state.resume == "Could not extract text from the PDF file.":
                    st.error("Unable to extract text. Please upload a text-based PDF.")

                else:
                    st.session_state.form_submitted = True
                    st.rerun()

if st.session_state.form_submitted:
    # Create an empty placeholder
    score_place = st.empty()

    # Display the loading message
    score_place.markdown(
        """
        <div style="background-color:#E3F2FD; padding:10px; border-radius:10px;">
            <span style="color:#1565C0; font-weight:bold;">
                Generating Scores...
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create a unique key for the current resume and job description
    cache_key = (st.session_state.resume,st.session_state.job_desc)     

    # Initialize report cache if it doesn't exist
    if "report_cache" not in st.session_state:
        st.session_state.report_cache = {}

    # Check whether the report is already available
    if cache_key in st.session_state.report_cache:

        # Reuse the previously generated report
        report = st.session_state.report_cache[cache_key]

        st.info("Using previously generated analysis report.")

    else:

        # Generate a new report using Groq
        report = get_report(
            st.session_state.resume,
            st.session_state.job_desc
        )

        # Cache the report for future use
        if report:
            st.session_state.report_cache[cache_key] = report

    # Stop further execution if report generation fails
    if report is None:
        score_place.empty()
        st.stop()

    # Calculate the Average Score from the LLM Report
    report_scores = extract_scores(report)

    if report_scores:
        avg_score = sum(report_scores) / len(report_scores)
        rating_display = f"{avg_score:.1f}/5"
    else:
        avg_score = None
        rating_display = "N/A"
        st.warning(
            "Unable to calculate the AI rating because "
            "no valid evaluation scores were generated."
        )
    # Call the function to get ATS Score
    ats_score = calculate_similarity_bert(st.session_state.resume,st.session_state.job_desc)
    
    col1,col2 = st.columns(2,border=True)
    with col1:
        st.markdown(
            "<div class='score-box'><b>🔗 Resume-Job Description Match: </b><br>"
            f"<span style='font-size:24px; color:#0047ab;'>{ats_score * 100:.0f}%</span><br>"
            "<small>Shows how closely your resume's content relates to the job description. This is an AI similarity estimate, not an actual ATS pass score.</small></div>",
            unsafe_allow_html=True,)

    with col2:
        st.markdown(
            "<div class='score-box'><b>📊 Overall AI Rating:</b><br>"
            f"<span style='font-size:24px; color:#228B22;'>{rating_display}</span><br>"
            "<small>Calculated from detailed analysis.</small></div>",
            unsafe_allow_html=True,
        )

        
    score_place.markdown(
    """
    <div style="background-color:#DFF6DD; padding:10px; border-radius:10px; margin-bottom:14px;">
        <span style="color:#2E7D32; font-weight:bold;">✅ Report generated!</span>
    </div>
    """,
    unsafe_allow_html=True
)


    st.markdown('<h4 style="color:#1a1a1a;">📋 Detailed Resume Evaluation Report</h4>', unsafe_allow_html=True)


    # Displaying Report 
    with st.container(border=True):
        st.markdown(report)
    
    # Download Button
    st.download_button(
        label="💾 Download Analysis Report",
        data=report,
        file_name="TalentIQ_Resume_Analysis.md",
        mime="text/markdown",
        icon=":material/download:",
    )
