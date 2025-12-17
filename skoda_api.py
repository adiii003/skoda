import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Store conversation sessions
sessions = {}

class SkodaChatbot:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.vector_store = None
        self.qa_chain = None
        self.support_number = "+91-1800-103-5000"
        self.support_email = "customercare@skoda.co.in"
        
        # Load models silently
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=200
        )
    
    def load_vector_store(self, path: str = "./skoda_faiss_index"):
        """Load FAISS index"""
        self.vector_store = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def setup_qa_chain(self, memory):
        """Setup QA chain with memory"""
        template = """You are Skoda's AI Customer Service Assistant. Your role is to help customers with their queries in a warm, professional, and helpful manner.

IMPORTANT GUIDELINES:
1. Keep responses SHORT and CONCISE (2-3 sentences maximum)
2. Be friendly, empathetic, and professional
3. Answer directly without unnecessary elaboration
4. Use the context provided to answer questions accurately
5. If you're unsure or the information isn't in the context, politely admit it and offer to escalate
6. For complex issues, service appointments, or emergencies, direct customers to call {support_number} or email {support_email}
7. Always maintain a conversational and natural tone

Context from Skoda documentation: {context}

Chat History: {chat_history}

Customer Question: {question}

Your Response (be warm, helpful, and BRIEF - max 2-3 sentences):""".format(
            support_number=self.support_number,
            support_email=self.support_email,
            context="{context}",
            chat_history="{chat_history}",
            question="{question}"
        )
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )
        
        return qa_chain
    
    def should_escalate(self, question: str, answer: str) -> bool:
        """Check if escalation needed"""
        escalation_keywords = [
            "don't know", "not sure", "can't help", "unable to", 
            "contact support", "call us"
        ]
        
        emergency_keywords = [
            "accident", "breakdown", "emergency", "book service", 
            "appointment", "test drive", "purchase", "buy"
        ]
        
        q_lower = question.lower()
        a_lower = answer.lower()
        
        if any(kw in q_lower for kw in emergency_keywords):
            return True
        if any(kw in a_lower for kw in escalation_keywords):
            return True
        
        return False


# Initialize chatbot globally
print("Initializing Skoda Chatbot...")

# Load API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in environment variables!")
    print("Please create a .env file with: GROQ_API_KEY=your_key_here")
    exit(1)

chatbot = SkodaChatbot(groq_api_key=GROQ_API_KEY)

# Check if vector store exists
if not os.path.exists("./skoda_faiss_index"):
    print("ERROR: FAISS vector store not found!")
    print("Please ensure 'skoda_faiss_index' folder exists in the same directory.")
    exit(1)

chatbot.load_vector_store("./skoda_faiss_index")
print("Chatbot Ready!\n")


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    
    Request JSON:
    {
        "message": "user message",
        "session_id": "unique_session_id"  // optional
    }
    
    Response JSON:
    {
        "reply": "bot response",
        "needs_escalation": true/false,
        "support_info": {
            "phone": "+91-1800-103-5000",
            "email": "customercare@skoda.co.in"
        }
    }
    """
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({
                "error": "Message is required"
            }), 400
        
        # Get or create session
        if session_id not in sessions:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            qa_chain = chatbot.setup_qa_chain(memory)
            sessions[session_id] = {
                "qa_chain": qa_chain,
                "memory": memory
            }
        
        # Get response
        qa_chain = sessions[session_id]["qa_chain"]
        result = qa_chain({"question": user_message})
        answer = result["answer"]
        
        # Check escalation
        needs_escalation = chatbot.should_escalate(user_message, answer)
        
        response = {
            "reply": answer,
            "needs_escalation": needs_escalation,
            "support_info": {
                "phone": chatbot.support_number,
                "email": chatbot.support_email
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "reply": "I'm having trouble processing your request. Please try again or contact our support team."
        }), 500


@app.route('/greeting', methods=['GET'])
def greeting():
    """
    Get welcome greeting
    
    Response JSON:
    {
        "message": "Welcome message..."
    }
    """
    greeting_msg = """Hello! 👋 Welcome to Skoda Customer Care!

I'm your AI assistant, here to help you with:
✨ Vehicle information and specifications
🔧 Service and maintenance queries
💡 Features and technology explanations
📋 Booking and support assistance
🚗 Model comparisons and recommendations

How may I assist you today?"""
    
    return jsonify({"message": greeting_msg}), 200


@app.route('/farewell', methods=['GET'])
def farewell():
    """
    Get farewell message
    
    Response JSON:
    {
        "message": "Thank you message..."
    }
    """
    farewell_msg = """Thank you for choosing Skoda! 🚗✨

It was a pleasure assisting you today. If you have any more questions in the 
future, I'm always here to help.

For immediate assistance, you can also reach us at:
📞 Call: +91-1800-103-5000
📧 Email: customercare@skoda.co.in

Drive safe and enjoy your Skoda experience! 🌟"""
    
    return jsonify({"message": farewell_msg}), 200


@app.route('/reset-session', methods=['POST'])
def reset_session():
    """
    Reset conversation session
    
    Request JSON:
    {
        "session_id": "unique_session_id"
    }
    """
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in sessions:
        del sessions[session_id]
    
    return jsonify({"message": "Session reset successfully"}), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Skoda Chatbot API"
    }), 200


if __name__ == '__main__':
    print("="*60)
    print("  SKODA CHATBOT API - RUNNING")
    print("="*60)
    print(f"  Server: http://localhost:5000")
    print(f"  Endpoints:")
    print(f"    POST /chat         - Send messages")
    print(f"    GET  /greeting     - Get welcome message")
    print(f"    GET  /farewell     - Get goodbye message")
    print(f"    POST /reset-session - Clear conversation")
    print(f"    GET  /health       - Check API status")
    print("="*60)
    

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
