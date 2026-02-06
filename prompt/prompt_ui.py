# import streamlit as st
# from huggingface_hub import InferenceClient
# from dotenv import load_dotenv

# import os

# # -------------------- ENV SETUP --------------------
# load_dotenv()

# hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# if not hf_token:
#     st.error("❌ HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")
#     st.stop()

# client = InferenceClient(api_key=hf_token)

# # -------------------- PAGE CONFIG --------------------
# st.set_page_config(
#     page_title="Research Paper Explainer",
#     layout="centered"
# )

# st.header("📄 Research Tool using HuggingFace")

# # -------------------- UI INPUTS --------------------

# paper_input = st.selectbox(
#     "Select Research Paper Name",
#     [
#         "Attention Is All You Need",
#         "BERT: Pre-training of Deep Bidirectional Transformers",
#         "GPT-3: Language Models are Few-Shot Learners",
#         "Diffusion Models Beat GANs on Image Synthesis"
#     ]
# )

# style_input = st.selectbox(
#     "Select Explanation Style",
#     [
#         "Beginner-Friendly",
#         "Technical",
#         "Code-Oriented",
#         "Mathematical"
#     ]
# )

# length_input = st.selectbox(
#     "Select Explanation Length",
#     [
#         "Short (1-2 paragraphs)",
#         "Medium (3-5 paragraphs)",
#         "Long (detailed explanation)"
#     ]
# )

# user_question = st.text_input(
#     "Enter your research question (optional):",
#     placeholder="Explain the core idea of this paper"
# )

# # -------------------- PROMPT BUILDER --------------------

# def build_prompt(paper, style, length, question):
#     base_prompt = f"""
# Explain the research paper titled "{paper}".

# Explanation style: {style}
# Explanation length: {length}
# """

#     if question:
#         base_prompt += f"\nUser question: {question}\n"

#     return base_prompt.strip()

# # -------------------- ACTION --------------------

# if st.button("Generate Explanation"):
#     with st.spinner("Generating explanation using HuggingFace..."):
#         prompt = build_prompt(
#             paper_input,
#             style_input,
#             length_input,
#             user_question
#         )

#         messages = [
#             {"role": "user", "content": prompt}
#         ]

#         result = client.chat_completion(
#             model="mistralai/Mistral-7B-Instruct-v0.2",
#             messages=messages,
#             max_tokens=800
#         )

#         st.subheader("📘 Generated Explanation")
#         st.write(result.choices[0].message.content)

# # -------------------- FOOTER --------------------
# st.markdown("---")
# st.caption("🚀 Powered by HuggingFace · Streamlit · Mistral-7B")





from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import os

load_dotenv()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(script_dir, 'template.json')

# Initialize HuggingFace InferenceClient
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    st.error("❌ HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables")
    st.stop()

client = InferenceClient(api_key=hf_token)

st.header('📚 Research Paper Explainer')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# Load prompt template
template = load_prompt(template_path)

if st.button('Generate Explanation'):
    with st.spinner("Generating explanation using HuggingFace..."):
        # Format the prompt
        prompt_text = template.format(
            paper_input=paper_input,
            style_input=style_input,
            length_input=length_input
        )
        
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        
        # Use chat completion API
        result = client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=messages,
            max_tokens=500,
            temperature=0.5
        )
        
        st.subheader("📖 Generated Explanation")
        st.write(result.choices[0].message.content)