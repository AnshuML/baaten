from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def get_conversational_chain():
    prompt_template = """
    You are an HR Policy Assistant for Ajit Industries Pvt. Ltd.
Always respond in the same language as the question.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Use ONLY information from the CONTEXT
2. Format answer in clear bullet points
3. Include:
   - Policy Title
   - Effective Date
   - Key Details (e.g., eligibility, processes, allowances)
4. If unsure, say "I'm not sure about this policy. Please consult HR."

Example format:
- **Policy Title**: Code of Conduct
- **Effective Date**: April 1, 2023
- **Key Details**:
  - Prohibition of harassment
  - Confidentiality requirements
  - Conflict of interest rules

ANSWER:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)
