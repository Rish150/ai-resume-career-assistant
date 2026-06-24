import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import re
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    groq_key = st.secrets.get("GROQ_API_KEY")

if not groq_key:
    st.write("Available secrets:", list(st.secrets.keys()))
    st.stop()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=groq_key
)
@st.cache_resource
def generate_interview_questions(vectorstore):

    docs = vectorstore.similarity_search(
        "skills projects experience certifications",
        k=8
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Generate 10 interview questions based ONLY on the
    information in the resume.

    Resume:
    {context}

    Rules:
    - Do not invent experience.
    - Do not invent projects.
    - Do not invent skills.
    - Tailor questions to the candidate's profile.

    For each question provide:
    1. Question
    2. Sample Answer
    3. Difficulty Level
    """

    response = llm.invoke(prompt)

    return response.content
def create_vectorstore(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(
        chunks,
        embeddings
    )
def ask_resume(question, vectorstore):

    docs = vectorstore.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are an expert AI Resume Assistant.

    Answer only using the provided resume information.

    If the answer is not present in the resume, respond with:
    "I could not find that information in the uploaded resume."

    Resume:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content
def match_resume_to_job(vectorstore,job_description):
    docs = vectorstore.similarity_search(job_description,k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Compare the candidate's resume with the job description.

    Resume:
    {context}

    Job Description:
    {job_description}

    Provide your response in EXACTLY this format:

    MATCH_SCORE: <number between 0 and 100>

    MATCHING_SKILLS:
    - skill1
    - skill2

    MISSING_SKILLS:
    - skill1
    - skill2

    STRENGTHS:
    - point1
    - point2

    SUGGESTIONS:
    - suggestion1
    - suggestion2

    Format the response clearly using bullet points.
    """
    response = llm.invoke(prompt)
    return response.content
def generate_cover_letter(
    vectorstore,
    job_description
):

    docs = vectorstore.similarity_search(
        job_description,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are an expert career coach.

    Write a professional cover letter using the
    candidate's resume and the job description.

    Resume:
    {context}

    Job Description:
    {job_description}

    Requirements:

    - Professional tone
    - Highlight relevant skills and experience
    - Explain why the candidate is a strong fit
    - Keep it concise (300-400 words)
    - Do not invent skills not present in the resume

    Cover Letter:
    """

    response = llm.invoke(prompt)

    return response.content
st.set_page_config(page_title = "AI Resume Assistant",page_icon = "🤖")
st.title("🤖 AI Resume Assistant")
st.write(
    """
    Upload your resume and ask questions about your experience,
    skills, projects, certifications, and job fit analysis.
    """
)
st.sidebar.title("Example questions")
st.sidebar.markdown("""
                - What skills do I have?
                - Summarize my Experience
                - What projects have I worked on?
                - Generate interview questions
                - What certifications do I have?
                - What are my strength
                    """)
generate_interview = st.sidebar.button(
    "Generate Interview Questions")
analyze_strengths = st.sidebar.button(
    "Analyze Resume Strengths"
)
cover_letter_btn = st.sidebar.button("Generate Cover Letter")
uploaded_file = st.file_uploader("Upload your Resume",type = "pdf")
job_description = st.text_area("Paste job description here (Optional)")
if uploaded_file:
    
    st.success("Resume uploaded succesfully")
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=".pdf")as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    vectorstore = create_vectorstore(
    pdf_path
)        
    question = st.text_input("Ask a question about your resume")
    if cover_letter_btn:

        if not job_description:

            st.warning(
                "Please paste a job description first."
            )

        else:

            with st.spinner(
                "Generating Cover Letter..."
            ):

                cover_letter = generate_cover_letter(
                    vectorstore,
                    job_description
                )

            st.markdown(
                "## Cover Letter"
            )

            st.markdown(
        "## Cover Letter"
    )

            st.text_area(
                "Generated Cover Letter",
                cover_letter,
                height=400
            )
    if st.button("Analyze Resume Match"):

        if job_description:

            with st.spinner(
                "Analyzing Resume Match..."
            ):

                result = match_resume_to_job(
                    vectorstore,
                    job_description
                )

            st.markdown("## ATS Analysis")

        match = re.search(
            r"MATCH_SCORE:\s*(\d+)",
            result
        )

        if match:

            score = int(
                match.group(1)
            )

            st.metric(
                label="ATS Match Score",
                value=f"{score}%"
            )

            st.progress(score)

            if score >= 80:

                st.success(
                    f"Excellent Match ({score}%)"
                )

            elif score >= 60:

                st.warning(
                    f"Good Match ({score}%)"
                )

            else:

                st.error(
                    f"Low Match ({score}%)"
                )

        with st.expander(
            "View Detailed Analysis"
        ):

            st.write(result)
            if st.button("Ask"):
                with st.spinner("Analysing Resume....."):
                    answer = ask_resume(
                    question,
                    vectorstore
                )
                st.markdown("### AI Response")
                st.success(answer)
        if generate_interview:

            with st.spinner(
                "Generating Interview Questions..."
            ):

                answer = generate_interview_questions(
                    vectorstore
                )

            st.markdown(
                "### Interview Questions"
            )

            st.success(answer)
        if analyze_strengths:

            with st.spinner(
                "Analyzing Resume..."
            ):

                answer = ask_resume(
                    """
                    Analyze the strengths and weaknesses of this resume.

                    Provide:
                    1. Key Strengths
                    2. Areas for Improvement
                    3. Missing Skills
                    4. Resume Suggestions
                    """,
                    vectorstore
                )

            st.markdown(
                "### Resume Analysis"
            )

            st.success(answer)
            