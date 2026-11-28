"""
RAG (Retrieval-Augmented Generation) Pipeline

Combines document retrieval with LLM generation.
Uses FAISS for efficient vector similarity search.
"""

from typing import List, Tuple, Optional, Dict
import os
import numpy as np
import pickle
import logging
import re
from .embeddings import EmbeddingsManager
from .llm_model import BaseLLM, create_llm

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
    faiss = None


def clean_section_content(text: Optional[str]) -> str:
    """
    Clean and normalize a section of LLM-generated roadmap content.

    Removes resume extraction artifacts, page markers, duplicated paragraphs/headings,
    and obvious raw-resume blocks such as "Resume Context:" or "CURRENT PROFILE:".

    Returns a trimmed, de-duplicated, professional block of text suitable for UI tabs.
    """
    if not text:
        return ""

    # Normalize newlines
    txt = text.replace('\r\n', '\n')

    # Remove explicit resume/context markers and their inline occurrence
    txt = re.sub(r'(?i)CURRENT\s*PROFILE\s*:\s*', '', txt)
    txt = re.sub(r'(?i)RESUME\s*CONTEXT\s*:\s*', '', txt)

    # Remove common page markers and separators
    txt = re.sub(r'(?m)^\s*Page\s+\d+(?:\s+of\s+\d+)?\s*$', '', txt)
    txt = re.sub(r'(?m)^[-*_]{3,}\s*$', '', txt)

    # Remove obvious extracted-resume header/footer lines
    txt = re.sub(r'(?im)^.*(EXTRACTED|RAW|OCR|EXTRACTION).*RESUME.*$', '', txt)
    txt = re.sub(r'(?im)^.*CURRENT\s+PROFILE.*$', '', txt)

    # Split into paragraphs and remove exact duplicates while preserving order
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', txt) if p.strip()]
    seen = set()
    unique_paras = []
    for p in paragraphs:
        key = re.sub(r'\s+', ' ', p).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique_paras.append(p)

    cleaned = '\n\n'.join(unique_paras)

    # Remove duplicated consecutive heading lines (case-insensitive)
    lines = cleaned.splitlines()
    out_lines = []
    prev_heading = None
    for ln in lines:
        stripped = ln.strip()
        if stripped:
            low = re.sub(r'\s+', ' ', stripped).lower()
            if prev_heading == low and len(stripped) < 200:
                # skip repeated heading-like line
                continue
            prev_heading = low
        out_lines.append(ln)

    cleaned = '\n'.join(out_lines).strip()

    # Final pass: remove lingering multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    return cleaned


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, dimension: int, save_path: str = "./vectorstore"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of embeddings
            save_path: Path to save/load index
        """
        if faiss is None:
            raise ImportError("FAISS is not installed")
        
        self.dimension = dimension
        self.save_path = save_path
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.documents = []  # Store actual documents
        self.metadata = []   # Store metadata
        
        os.makedirs(save_path, exist_ok=True)
    
    def add(self, embeddings: np.ndarray, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add vectors and documents to index.
        
        Args:
            embeddings: Array of embeddings
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings and documents must match")
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        if metadata is None:
            metadata = [{} for _ in documents]
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document, distance, metadata) tuples
        """
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distance)
                results.append((
                    self.documents[int(idx)],
                    float(similarity),
                    self.metadata[int(idx)]
                ))
        
        return results
    
    def save(self, index_path: Optional[str] = None, docs_path: Optional[str] = None):
        """Save index and documents."""
        index_path = index_path or os.path.join(self.save_path, "faiss_index")
        docs_path = docs_path or os.path.join(self.save_path, "documents.pkl")
        
        faiss.write_index(self.index, index_path)
        
        with open(docs_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"Saved FAISS index and documents")
    
    def load(self, index_path: Optional[str] = None, docs_path: Optional[str] = None):
        """Load index and documents."""
        index_path = index_path or os.path.join(self.save_path, "faiss_index")
        docs_path = docs_path or os.path.join(self.save_path, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            
            logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
            return True
        
        return False
    
    def __len__(self):
        """Get number of documents in store."""
        return len(self.documents)


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline.
    
    Combines document retrieval with LLM generation.
    """
    
    def __init__(
        self,
        embeddings_model: str = "all-MiniLM-L6-v2",
        llm_type: str = "ollama",
        llm_model: str = "tinyllama",
        vectorstore_path: str = "./vectorstore"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embeddings_model: Name of embedding model
            llm_type: Type of LLM ('ollama' or 'openai')
            llm_model: Name of LLM model (default: tinyllama for CPU stability)
            vectorstore_path: Path to store vector index
        """
        self.embeddings_manager = EmbeddingsManager(embeddings_model)
        self.embedding_dim = self.embeddings_manager.get_embedding_dimension()
        
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_dim,
            save_path=vectorstore_path
        )
        
        self.llm = create_llm(llm_type=llm_type, model_name=llm_model)
        self.vectorstore_path = vectorstore_path
        
        logger.info(f"RAG Pipeline initialized with LLM: {llm_model} (type: {llm_type})")
        
        logger.info("RAG Pipeline initialized successfully")
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document chunks
            metadata: Optional metadata for tracking sources
        """
        if not documents:
            logger.warning("No documents provided")
            return
        
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embeddings_manager.generate_embeddings(documents)
        
        self.vector_store.add(embeddings, documents, metadata)
        logger.info(f"Successfully added {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score, metadata) tuples
        """
        query_embedding = self.embeddings_manager.generate_single_embedding(query)
        results = self.vector_store.search(query_embedding, k=top_k)
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate answer.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            temperature: Temperature for LLM generation
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for query")
            return {
                "answer": "No relevant information found in knowledge base.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Step 2: Prepare context
        context_text = "\n\n".join([doc for doc, _, _ in retrieved_docs])
        sources = [meta.get('source', 'unknown') for _, _, meta in retrieved_docs]
        
        # Step 3: Generate answer using LLM
        answer = self.llm.generate_with_context(
            context=context_text,
            question=query,
            temperature=temperature
        )
        
        # Step 4: Calculate confidence (average similarity of top documents)
        confidence = np.mean([score for _, score, _ in retrieved_docs])
        
        logger.info(f"Generated answer with confidence: {confidence:.2f}")
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": float(confidence),
            "retrieved_docs": len(retrieved_docs)
        }
    
    def save_index(self) -> None:
        """Save the vector index to disk."""
        self.vector_store.save()
        logger.info("Vector index saved to disk")
    
    def load_index(self) -> bool:
        """Load vector index from disk."""
        success = self.vector_store.load()
        if success:
            logger.info("Vector index loaded from disk")
        return success
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        return {
            "total_documents": len(self.vector_store),
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.embeddings_manager.model_name,
            "vectorstore_path": self.vectorstore_path
        }
    
    def analyze_resume(self, resume_text: str, temperature: float = 0.7) -> Dict:
        """
        Analyze resume using LLM with structured prompt.
        
        Args:
            resume_text: Extracted resume text
            temperature: Temperature for LLM generation
            
        Returns:
            Dictionary with structured resume analysis
        """
        logger.info("Starting resume analysis...")
        
        # Structured prompt for resume analysis
        analysis_prompt = f"""You are an expert ATS resume reviewer, HR specialist, and AI career mentor.

Analyze the following resume carefully and provide a comprehensive analysis with these exact sections:

## 1. TECHNICAL SKILLS
List all technical skills identified in the resume (programming languages, tools, frameworks, platforms, certifications)

## 2. CAREER STRENGTHS
Identify the key strengths of this resume:
- What makes this candidate stand out?
- What are their strongest qualifications?
- What is their domain expertise?

## 3. WEAKNESSES & MISSING AREAS
Identify gaps and areas for improvement:
- What important skills are missing?
- What experience gaps exist?
- What would make this candidate more competitive?

## 4. ATS (Applicant Tracking System) OPTIMIZATION TIPS
Provide specific, actionable suggestions to improve ATS compatibility:
- Which keywords should be added?
- How can the formatting be improved?
- Which certifications or metrics should be highlighted?

## 5. CAREER ROLE RECOMMENDATIONS
Based on the resume, recommend 3-5 specific job roles/positions that would be a good fit

## 6. RECOMMENDED LEARNING PATH
Suggest a learning roadmap:
- What skills to acquire next?
- What certifications would be valuable?
- Estimated timeframe for each milestone?

## 7. INTERVIEW PREPARATION TIPS
Provide 5-7 specific tips to prepare for interviews based on this resume:
- Key talking points
- Potential interview questions
- How to frame past experiences

---

RESUME:
{resume_text}

---

Provide detailed, practical, and actionable analysis. Be specific and professional."""

        try:
            # Generate analysis using LLM
            analysis_text = self.llm.generate(
                prompt=analysis_prompt,
                temperature=temperature
            )
            
            # Parse response into sections
            sections = self._parse_resume_analysis(analysis_text)
            
            logger.info("Resume analysis completed successfully")
            
            return {
                "success": True,
                "full_analysis": analysis_text,
                "sections": sections,
                "resume_length_chars": len(resume_text)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing resume: {e}")
            return {
                "success": False,
                "error": str(e),
                "full_analysis": None,
                "sections": {}
            }
    
    def _parse_resume_analysis(self, analysis_text: str) -> Dict[str, str]:
        """
        Parse resume analysis text into structured sections.
        
        Args:
            analysis_text: Full analysis text from LLM
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Define section headers to look for
        section_headers = [
            ("Technical Skills", "## 1. TECHNICAL SKILLS"),
            ("Strengths", "## 2. CAREER STRENGTHS"),
            ("Weaknesses", "## 3. WEAKNESSES & MISSING AREAS"),
            ("ATS Suggestions", "## 4. ATS (APPLICANT TRACKING SYSTEM) OPTIMIZATION TIPS"),
            ("Career Recommendations", "## 5. CAREER ROLE RECOMMENDATIONS"),
            ("Learning Path", "## 6. RECOMMENDED LEARNING PATH"),
            ("Interview Tips", "## 7. INTERVIEW PREPARATION TIPS")
        ]
        
        # Try to extract sections
        for section_name, header in section_headers:
            try:
                if header in analysis_text:
                    # Find content after header
                    start_idx = analysis_text.find(header)
                    start_idx = analysis_text.find("\n", start_idx) + 1
                    
                    # Find next section header or end of text
                    next_section_idx = len(analysis_text)
                    for _, next_header in section_headers:
                        if next_header != header:
                            next_idx = analysis_text.find(next_header, start_idx)
                            if next_idx != -1 and next_idx < next_section_idx:
                                next_section_idx = next_idx
                    
                    sections[section_name] = analysis_text[start_idx:next_section_idx].strip()
                else:
                    sections[section_name] = ""
            except:
                sections[section_name] = ""
        
        return sections
    
    def calculate_ats_score(self, resume_text: str) -> Dict:
        """
        Calculate a simple ATS compatibility score (0-100).
        
        Args:
            resume_text: Resume text to analyze
            
        Returns:
            Dictionary with score and breakdown
        """
        import re
        
        score = 0
        breakdown = {}
        
        # Normalize text
        resume_lower = resume_text.lower()
        
        # 1. Check for key technical sections (25 points)
        section_keywords = {
            "skills": ["skills", "technical skills", "competencies", "expertise"],
            "experience": ["experience", "work experience", "professional experience"],
            "education": ["education", "degree", "university", "college"],
            "projects": ["projects", "portfolio", "achievements", "accomplishments"]
        }
        
        sections_found = 0
        for section, keywords in section_keywords.items():
            if any(kw in resume_lower for kw in keywords):
                sections_found += 1
        
        section_score = min(25, sections_found * 6)
        breakdown["sections"] = section_score
        score += section_score
        
        # 2. Check for common keywords (25 points)
        tech_keywords = [
            "python", "java", "javascript", "sql", "api", "database",
            "aws", "cloud", "docker", "kubernetes", "git", "agile",
            "machine learning", "data science", "frontend", "backend",
            "full stack", "devops", "testing", "ci/cd"
        ]
        
        keywords_found = sum(1 for kw in tech_keywords if kw in resume_lower)
        keyword_score = min(25, keywords_found * 2)
        breakdown["keywords"] = keyword_score
        score += keyword_score
        
        # 3. Check for certifications/achievements (15 points)
        cert_keywords = [
            "certified", "aws", "azure", "gcp", "scrum", "pmp",
            "honors", "award", "first", "published", "patent"
        ]
        
        if any(cert in resume_lower for cert in cert_keywords):
            breakdown["certifications"] = 15
            score += 15
        else:
            breakdown["certifications"] = 0
        
        # 4. Check for quantifiable metrics (15 points)
        # Look for numbers and percentages
        if re.search(r'\d+%', resume_text) or re.search(r'\$[\d,]+', resume_text):
            breakdown["metrics"] = 15
            score += 15
        else:
            breakdown["metrics"] = 0
        
        # 5. Check for formatting and length (10 points)
        word_count = len(resume_text.split())
        
        # Good ATS resumes are typically 250-1000 words
        if 250 <= word_count <= 1000:
            breakdown["formatting"] = 10
            score += 10
        elif word_count > 100:
            breakdown["formatting"] = 5
            score += 5
        else:
            breakdown["formatting"] = 0
        
        # 5. Check for action verbs (10 points)
        action_verbs = [
            "developed", "designed", "implemented", "created", "built",
            "led", "managed", "improved", "optimized", "achieved",
            "launched", "collaborated", "contributed", "increased", "reduced"
        ]
        
        action_verb_count = sum(1 for verb in action_verbs if verb in resume_lower)
        verb_score = min(10, action_verb_count)
        breakdown["action_verbs"] = verb_score
        score += verb_score
        
        # Ensure score is between 0-100
        final_score = min(100, max(0, score))
        
        return {
            "ats_score": final_score,
            "score_breakdown": breakdown,
            "word_count": word_count,
            "recommendation": self._get_ats_recommendation(final_score)
        }
    
    def _get_ats_recommendation(self, score: int) -> str:
        """
        Get ATS recommendation based on score.
        
        Args:
            score: ATS score
            
        Returns:
            Recommendation text
        """
        if score >= 80:
            return "✅ Excellent! Your resume is well-optimized for ATS systems."
        elif score >= 70:
            return "✓ Good! Your resume should pass most ATS filters with minor improvements."
        elif score >= 60:
            return "⚠️ Fair. Consider implementing some ATS optimization suggestions."
        elif score >= 50:
            return "⚠️ Needs improvement. Follow the ATS suggestions to increase your score."
        else:
            return "❌ Poor. Major revisions recommended to improve ATS compatibility."
    
    def extract_skills(self, text: str) -> Dict:
        """
        Extract technical and soft skills from text using keyword matching.
        
        Args:
            text: Input text (resume or job description)
            
        Returns:
            Dictionary with extracted skills
        """
        logger.info("Extracting skills from text using keyword matching...")
        
        # Define comprehensive skill keywords
        technical_keywords = [
            # Programming Languages
            "python", "java", "javascript", "typescript", "c++", "csharp", "c#", "golang", "go", 
            "rust", "kotlin", "swift", "ruby", "php", "r", "scala", "perl", "matlab", "vb",
            "sql", "plsql", "tsql",
            
            # Web Frameworks & Libraries
            "react", "angular", "vue", "vue.js", "express", "django", "flask", "fastapi",
            "springboot", "spring", "asp.net", "aspnet", "node.js", "nodejs",
            "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy",
            "matplotlib", "seaborn", "plotly", "scipy",
            
            # Databases
            "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql", "redis", "dynamodb",
            "cassandra", "elasticsearch", "oracle", "mssql", "firestore", "supabase",
            
            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "github",
            "circleci", "terraform", "ansible", "cloudformation",
            
            # AI/ML & Data Science
            "machine learning", "deep learning", "artificial intelligence", "ai", "nlp",
            "natural language processing", "computer vision", "cv", "tensorflow", "pytorch",
            "keras", "scikit-learn", "pandas", "numpy", "rag", "llm", "nlp",
            
            # Vision & Image Processing
            "opencv", "mediapipe", "yolo", "cnn", "convolutional", "resnet", "vgg",
            "inception", "mobilenet", "transformers",
            
            # Web Technologies
            "html", "css", "xml", "json", "rest", "graphql", "websocket", "http", "https",
            "api", "rest api", "grpc",
            
            # Tools & Platforms
            "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
            "docker", "kubernetes", "helm", "jenkins", "circleci", "gitlab ci",
            "streamlit", "jupyter", "anaconda", "pip", "npm", "yarn", "maven", "gradle",
            "vim", "vscode", "intellij", "pycharm",
            
            # Soft Skills Technical
            "linux", "unix", "windows", "macos", "bash", "shell", "powershell",
            "agile", "scrum", "kanban", "ci/cd", "devops", "monitoring",
            "microservices", "monolithic", "serverless", "lambda",
        ]
        
        soft_keywords = [
            "communication", "teamwork", "leadership", "collaboration", "problem-solving",
            "critical thinking", "creativity", "adaptability", "management", "project management",
            "time management", "negotiation", "presentation", "writing", "analytical",
            "detail-oriented", "organization", "mentoring", "coaching", "delegation",
            "conflict resolution", "empathy", "emotional intelligence", "motivation",
        ]
        
        certification_keywords = [
            "aws certified", "azure certified", "gcp certified", "google cloud",
            "aws solutions", "aws developer", "aws devops", "aws sysops",
            "azure fundamentals", "azure administrator", "azure developer",
            "gcp associate", "gcp professional",
            "kubernetes", "ckad", "cka",
            "cissp", "security+", "sec+",
            "pmp", "capm", "prince2", "scrum master", "csm",
            "ccna", "ccnp", "comptia", "linuxplus",
            "certification", "certified", "exam",
        ]
        
        # Normalize text
        normalized_text = text.lower()
        # Remove special characters and extra spaces
        normalized_text = re.sub(r'[^\w\s]', ' ', normalized_text)
        normalized_text = re.sub(r'\s+', ' ', normalized_text)
        
        logger.info(f"Normalized text length: {len(normalized_text)} chars")
        
        # Extract matching skills
        found_technical = []
        found_soft = []
        found_certifications = []
        
        # Match technical skills
        for skill in technical_keywords:
            if skill in normalized_text:
                found_technical.append(skill)
                logger.debug(f"Found technical skill: {skill}")
        
        # Match soft skills
        for skill in soft_keywords:
            if skill in normalized_text:
                found_soft.append(skill)
                logger.debug(f"Found soft skill: {skill}")
        
        # Match certifications
        for cert in certification_keywords:
            if cert in normalized_text:
                found_certifications.append(cert)
                logger.debug(f"Found certification: {cert}")
        
        # Remove duplicates while preserving order
        found_technical = list(dict.fromkeys(found_technical))
        found_soft = list(dict.fromkeys(found_soft))
        found_certifications = list(dict.fromkeys(found_certifications))
        
        result = {
            "technical": found_technical,
            "soft": found_soft,
            "certifications": found_certifications,
            "domain": []  # Not used in keyword matching
        }
        
        logger.info(f"Extracted {len(found_technical)} technical, {len(found_soft)} soft, {len(found_certifications)} certification skills")
        return result
    
    
    def compare_skills(self, resume_skills: Dict, jd_skills: Dict) -> Dict:
        """
        Compare skills between resume and job description.
        Uses exact and fuzzy matching for keyword-based skill extraction.
        
        Args:
            resume_skills: Extracted skills from resume
            jd_skills: Extracted skills from job description
            
        Returns:
            Dictionary with matching and missing skills
        """
        logger.info("Comparing skills between resume and job description...")
        
        def normalize_skill(skill: str) -> str:
            """Normalize skill name for comparison."""
            normalized = skill.lower().strip()
            normalized = re.sub(r'[^\w\s]', ' ', normalized)
            normalized = re.sub(r'\s+', ' ', normalized)
            return normalized
        
        def find_similar_skills(skill: str, skill_list: List[str], threshold: float = 0.7) -> List[str]:
            """Find similar skills using fuzzy matching."""
            from difflib import SequenceMatcher
            
            normalized_skill = normalize_skill(skill)
            matches = []
            
            for other_skill in skill_list:
                normalized_other = normalize_skill(other_skill)
                
                # Exact match (after normalization)
                if normalized_skill == normalized_other:
                    matches.append(other_skill)
                    continue
                
                # Substring match (one contains the other)
                if normalized_skill in normalized_other or normalized_other in normalized_skill:
                    matches.append(other_skill)
                    continue
                
                # Fuzzy match using SequenceMatcher
                similarity = SequenceMatcher(None, normalized_skill, normalized_other).ratio()
                if similarity >= threshold:
                    matches.append(other_skill)
            
            return matches
        
        result = {
            "matching_technical": [],
            "missing_technical": [],
            "matching_soft": [],
            "missing_soft": [],
            "matching_certifications": [],
            "missing_certifications": [],
            "resume_unique": [],
            "jd_unique": []
        }
        
        # Technical skills comparison
        logger.debug(f"Resume technical skills: {resume_skills.get('technical', [])}")
        logger.debug(f"JD technical skills: {jd_skills.get('technical', [])}")
        
        for skill in jd_skills.get("technical", []):
            matches = find_similar_skills(skill, resume_skills.get("technical", []))
            if matches:
                result["matching_technical"].append(skill)
                logger.debug(f"Technical skill match: {skill}")
            else:
                result["missing_technical"].append(skill)
                logger.debug(f"Technical skill missing: {skill}")
        
        # Resume technical skills not in JD
        for skill in resume_skills.get("technical", []):
            matches = find_similar_skills(skill, jd_skills.get("technical", []))
            if not matches:
                result["resume_unique"].append(skill)
        
        # Soft skills comparison
        for skill in jd_skills.get("soft", []):
            matches = find_similar_skills(skill, resume_skills.get("soft", []))
            if matches:
                result["matching_soft"].append(skill)
                logger.debug(f"Soft skill match: {skill}")
            else:
                result["missing_soft"].append(skill)
                logger.debug(f"Soft skill missing: {skill}")
        
        # Resume soft skills not in JD
        for skill in resume_skills.get("soft", []):
            matches = find_similar_skills(skill, jd_skills.get("soft", []))
            if not matches:
                result["resume_unique"].append(skill)
        
        # Certifications comparison
        for cert in jd_skills.get("certifications", []):
            matches = find_similar_skills(cert, resume_skills.get("certifications", []))
            if matches:
                result["matching_certifications"].append(cert)
                logger.debug(f"Certification match: {cert}")
            else:
                result["missing_certifications"].append(cert)
                logger.debug(f"Certification missing: {cert}")
        
        matching_count = (
            len(result["matching_technical"]) +
            len(result["matching_soft"]) +
            len(result["matching_certifications"])
        )
        missing_count = (
            len(result["missing_technical"]) +
            len(result["missing_soft"]) +
            len(result["missing_certifications"])
        )
        
        logger.info(f"Skill comparison complete:")
        logger.info(f"  Matching: {matching_count}")
        logger.info(f"  Missing: {missing_count}")
        logger.info(f"  Resume unique: {len(result['resume_unique'])}")
        
        return result
    
    def calculate_match_percentage(self, skill_comparison: Dict, resume_text: str, jd_text: str) -> float:
        """
        Calculate overall match percentage based on skills and content similarity.
        
        Score breakdown:
        - 70% based on skill matching
        - 30% based on text content similarity
        
        Args:
            skill_comparison: Result from compare_skills()
            resume_text: Resume text
            jd_text: Job description text
            
        Returns:
            Match percentage (0-100)
        """
        logger.info("Calculating match percentage...")
        
        # Count all job description skills
        total_jd_skills = (
            len(skill_comparison.get("matching_technical", [])) +
            len(skill_comparison.get("missing_technical", [])) +
            len(skill_comparison.get("matching_soft", [])) +
            len(skill_comparison.get("missing_soft", [])) +
            len(skill_comparison.get("matching_certifications", [])) +
            len(skill_comparison.get("missing_certifications", []))
        )
        
        # Count matched skills
        matched_skills = (
            len(skill_comparison.get("matching_technical", [])) +
            len(skill_comparison.get("matching_soft", [])) +
            len(skill_comparison.get("matching_certifications", []))
        )
        
        logger.info(f"Total JD skills: {total_jd_skills}, Matched: {matched_skills}")
        
        # If no skills found in JD, use text similarity only
        if total_jd_skills == 0:
            logger.warning("No skills found in job description")
            # Simple text overlap calculation
            resume_words = set(re.findall(r'\w+', resume_text.lower()))
            jd_words = set(re.findall(r'\w+', jd_text.lower()))
            if len(jd_words) == 0:
                return 0.0
            overlap = len(resume_words & jd_words)
            return min(100.0, (overlap / len(jd_words)) * 100)
        
        # Calculate skills-based score (70% weight)
        skills_match_ratio = matched_skills / total_jd_skills
        skills_score = skills_match_ratio * 70
        
        # Calculate text similarity score (30% weight)
        resume_words = set(re.findall(r'\w+', resume_text.lower()))
        jd_words = set(re.findall(r'\w+', jd_text.lower()))
        
        if len(jd_words) == 0:
            text_similarity_score = 0
        else:
            overlap = len(resume_words & jd_words)
            text_similarity_ratio = overlap / len(jd_words)
            text_similarity_score = text_similarity_ratio * 30
        
        total_score = skills_score + text_similarity_score
        final_score = min(100.0, max(0.0, total_score))
        
        logger.info(f"Match calculation:")
        logger.info(f"  Skills score (70%): {skills_score:.1f}")
        logger.info(f"  Text similarity (30%): {text_similarity_score:.1f}")
        logger.info(f"  Final score: {final_score:.1f}%")
        
        return round(final_score, 1)
    
    def generate_job_recommendations(
        self, 
        resume_text: str, 
        jd_text: str, 
        skill_comparison: Dict,
        match_percentage: float,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate AI recommendations for improving job match.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
            skill_comparison: Result from compare_skills()
            match_percentage: Match percentage from calculate_match_percentage()
            temperature: Temperature for LLM
            
        Returns:
            Dictionary with recommendations
        """
        logger.info("Generating job match recommendations...")
        
        missing_skills = skill_comparison.get("missing_technical", [])[:5]  # Top 5
        missing_soft = skill_comparison.get("missing_soft", [])[:3]  # Top 3
        
        recommendation_prompt = f"""You are an expert career coach and resume optimizer.

Based on this job description and resume analysis, provide specific recommendations:

JOB DESCRIPTION (EXCERPT):
{jd_text[:1500]}

CURRENT MATCH SCORE: {match_percentage:.1f}%

MISSING TECHNICAL SKILLS: {', '.join(missing_skills) if missing_skills else 'None identified'}
MISSING SOFT SKILLS: {', '.join(missing_soft) if missing_soft else 'None identified'}

Please provide:

## IMPROVEMENT SUGGESTIONS
- 3-4 specific ways to strengthen this resume for THIS job
- Focus on the most impactful changes

## RECOMMENDED PROJECTS
- 2-3 specific projects the candidate should build/complete
- Include estimated timeframe and why each matters for this role
- Be specific about tech stack

## RECOMMENDED CERTIFICATIONS
- 2-3 certifications that would boost candidacy
- Why each is relevant to this specific job

## RESUME OPTIMIZATION
- 3-4 specific changes to make to the resume
- What keywords/phrases to add
- How to reframe existing experience

## HIRING PROBABILITY ASSESSMENT
Based on the {match_percentage:.1f}% match and missing skills:
- Is this candidate likely to get shortlisted? (High/Medium/Low)
- What's the realistic next step for this candidate?
- How long would it take to become a strong candidate?

Be specific, actionable, and realistic."""

        try:
            response = self.llm.generate(prompt=recommendation_prompt, temperature=temperature)
            
            sections = self._parse_recommendations(response)
            
            logger.info("Recommendations generated successfully")
            return {
                "success": True,
                "full_recommendations": response,
                "sections": sections
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                "success": False,
                "error": str(e),
                "full_recommendations": None,
                "sections": {}
            }
    
    def _parse_recommendations(self, recommendations_text: str) -> Dict[str, str]:
        """
        Parse recommendations text into structured sections.
        
        Args:
            recommendations_text: Full recommendations text from LLM
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        section_headers = [
            ("Improvement Suggestions", "## IMPROVEMENT SUGGESTIONS"),
            ("Recommended Projects", "## RECOMMENDED PROJECTS"),
            ("Recommended Certifications", "## RECOMMENDED CERTIFICATIONS"),
            ("Resume Optimization", "## RESUME OPTIMIZATION"),
            ("Hiring Probability", "## HIRING PROBABILITY ASSESSMENT")
        ]
        
        for section_name, header in section_headers:
            try:
                if header in recommendations_text:
                    start_idx = recommendations_text.find(header)
                    start_idx = recommendations_text.find("\n", start_idx) + 1
                    
                    next_section_idx = len(recommendations_text)
                    for _, next_header in section_headers:
                        if next_header != header:
                            next_idx = recommendations_text.find(next_header, start_idx)
                            if next_idx != -1 and next_idx < next_section_idx:
                                next_section_idx = next_idx
                    
                    sections[section_name] = recommendations_text[start_idx:next_section_idx].strip()
                else:
                    sections[section_name] = ""
            except:
                sections[section_name] = ""
        
        return sections
    
    def analyze_job_match(
        self,
        resume_text: str,
        jd_text: str,
        temperature: float = 0.7
    ) -> Dict:
        """
        Complete job matching pipeline.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
            temperature: Temperature for LLM
            
        Returns:
            Complete job matching analysis
        """
        logger.info("Starting complete job match analysis...")
        
        try:
            # Step 1: Extract skills from both
            resume_skills = self.extract_skills(resume_text)
            jd_skills = self.extract_skills(jd_text)
            
            # Step 2: Compare skills
            skill_comparison = self.compare_skills(resume_skills, jd_skills)
            
            # Step 3: Calculate match percentage
            match_percentage = self.calculate_match_percentage(
                skill_comparison, resume_text, jd_text
            )
            
            # Step 4: Generate recommendations
            recommendations = self.generate_job_recommendations(
                resume_text, jd_text, skill_comparison, match_percentage, temperature
            )
            
            logger.info(f"Job match analysis complete. Match: {match_percentage:.1f}%")
            
            return {
                "success": True,
                "match_percentage": match_percentage,
                "resume_skills": resume_skills,
                "jd_skills": jd_skills,
                "skill_comparison": skill_comparison,
                "recommendations": recommendations,
                "recommendation_sections": recommendations.get("sections", {})
            }
        
        except Exception as e:
            logger.error(f"Error in job match analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "match_percentage": 0.0
            }
    
    def generate_career_roadmap(
        self,
        resume_text: str,
        career_goal: str,
        experience_level: str,
        timeline: str,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate a personalized AI career roadmap based on resume and goals.
        
        Args:
            resume_text: Extracted resume text
            career_goal: Target career role (e.g., "AI Engineer", "Data Scientist")
            experience_level: Current level (Beginner, Intermediate, Advanced)
            timeline: Timeline for achievement (30 Days, 3 Months, 6 Months, 1 Year)
            temperature: Temperature for LLM generation
            
        Returns:
            Dictionary with personalized career roadmap
        """
        logger.info(f"Generating career roadmap for goal: {career_goal}")
        
        try:
            # Extract current skills from resume
            resume_skills = self.extract_skills(resume_text)
            
            # Get all skills for reference
            all_technical = resume_skills.get("technical", [])
            all_soft = resume_skills.get("soft", [])
            all_certs = resume_skills.get("certifications", [])
            
            # Create comprehensive roadmap prompt
            roadmap_prompt = f"""You are an expert AI/tech career coach and mentor. Based on the provided information, create a DETAILED and PERSONALIZED career roadmap.

IMPORTANT: Return ONLY the following sections, in the exact order shown, using the exact ALL CAPS labels followed immediately by a colon. Do not include any other headings, emojis, or extra section names. Do not repeat content across sections. Do not include any raw resume text.

CAREER_GOAL:
{career_goal}

SKILL_ANALYSIS:
- Existing Strengths (bullet points)
- Weak Areas (bullet points)
- Skills Gap (bullet points)

LEARNING_ROADMAP:
- Detailed milestone-based plan for the timeline ({timeline})
- Weekly milestones for 30 days and 3 months, monthly for 6 months, quarterly for 1 year
- Include duration, key topics, tools, and expected outcomes

RECOMMENDED_PROJECTS:
- 3-5 project ideas tailored to the target role and experience level
- For each project, include name, relevance, tech stack, and where to showcase

LEARNING_RESOURCES:
- Courses
- Certifications
- Books
- Websites
- Channels

INTERVIEW_PREPARATION:
- Key topics
- Sample questions
- System design notes
- Behavioral tips

CAREER_STRATEGY:
- GitHub strategy
- LinkedIn strategy
- Networking and internship strategy

KEY_MILESTONES:
- Month-by-month or quarter-by-quarter milestones to achieve {career_goal} by {timeline}

Do not include raw resume excerpts, page markers, or extracted resume artifacts.
"""

            # Generate roadmap using LLM
            response = self.llm.generate(prompt=roadmap_prompt, temperature=temperature)
            
            # Parse sections from response
            sections = self._parse_roadmap_sections(response)
            # Clean the raw full roadmap before returning to UI (remove resume dumps)
            cleaned_full_roadmap = clean_section_content(response)
            
            logger.info(f"Career roadmap generated successfully for {career_goal}")
            
            # Add top-level section fields for easier consumption
            return {
                "success": True,
                "career_goal": career_goal,
                "experience_level": experience_level,
                "timeline": timeline,
                "full_roadmap": cleaned_full_roadmap,
                "sections": sections,
                "skill_analysis": sections.get("Skill Analysis", ""),
                "learning_roadmap": sections.get("Learning Roadmap", ""),
                "recommended_projects": sections.get("Recommended Projects", ""),
                "learning_resources": sections.get("Learning Resources", ""),
                "interview_preparation": sections.get("Interview Preparation", ""),
                "career_strategy": sections.get("Career Strategy", ""),
                "key_milestones": sections.get("Key Milestones", "")
            }
        
        except Exception as e:
            logger.error(f"Error generating career roadmap: {e}")
            return {
                "success": False,
                "error": str(e),
                "career_goal": career_goal,
                "experience_level": experience_level,
                "timeline": timeline,
                "full_roadmap": None,
                "sections": {}
            }
    
    def _parse_roadmap_sections(self, roadmap_text: str) -> Dict[str, str]:
        """
        Parse career roadmap text into structured sections.
        
        Args:
            roadmap_text: Full roadmap text from LLM
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        normalized_text = (roadmap_text or "").replace("\r\n", "\n").strip()
        label_map = {
            "CAREER_GOAL": "Career Goal",
            "SKILL_ANALYSIS": "Skill Analysis",
            "LEARNING_ROADMAP": "Learning Roadmap",
            "RECOMMENDED_PROJECTS": "Recommended Projects",
            "LEARNING_RESOURCES": "Learning Resources",
            "INTERVIEW_PREPARATION": "Interview Preparation",
            "CAREER_STRATEGY": "Career Strategy",
            "KEY_MILESTONES": "Key Milestones"
        }

        alternation = "|".join(label_map.keys())
        pattern = re.compile(
            rf"(?is)^\s*(?P<label>{alternation})\s*:\s*(?P<content>.*?)(?=^\s*(?:{alternation})\s*:|\Z)",
            flags=re.MULTILINE | re.DOTALL
        )

        extracted = {v: "" for v in label_map.values()}
        for match in pattern.finditer(normalized_text):
            label = match.group("label").upper()
            content = match.group("content").strip()
            section_name = label_map[label]
            extracted[section_name] = clean_section_content(content)

        for section_name, section_content in extracted.items():
            sections[section_name] = section_content
            logger.debug(
                "Roadmap section parsed: %s found=%s length=%d",
                section_name,
                bool(section_content),
                len(section_content)
            )

        duplicates = {}
        for section_name, section_content in sections.items():
            if not section_content:
                continue
            normalized_section = re.sub(r"\s+", " ", section_content).strip()
            duplicates.setdefault(normalized_section, []).append(section_name)

        duplicate_groups = [names for names in duplicates.values() if len(names) > 1]
        if duplicate_groups:
            logger.warning(
                "Roadmap duplicate content detected in sections: %s",
                duplicate_groups
            )

        return sections
