import fitz  # extract text
import torch #deep 
import re
from transformers import AutoTokenizer, AutoModel #pre trained models
from transformers import pipeline
from sklearn.cluster import KMeans #clustering data 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PaperProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter', timeout=60) #good for science papers. we need the citation graph
        self.model = AutoModel.from_pretrained('allenai/specter')
        

    #self explanatoyr
    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def extract_bold_headers(pdf_path):
        doc = fitz.open(pdf_path)
        bold_headers = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_instances = page.extract_text("dict")  # Extract text in dictionary format
            
            # Loop through text instances
            for block in text_instances["blocks"]:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Check if the span's font is bold
                        if "bold" in span["font"].lower():  # Check for the word "bold" in the font name
                            bold_headers.append(span["text"])

        return bold_headers
    
    def identify_sections(self, text, pdf_path=None):
        # Using regex patterns for common section headers
        section_patterns = {
            'abstract': r'abstract.*?\n',
            'introduction': r'introduction.*?\n',
            'methodology': r'(methodology|methods).*?\n',
            'results': r'results.*?\n',
            'discussion': r'discussion.*?\n',
            'conclusion': r'conclusion.*?\n',
            # 'key_words': r'key words.*?\n',
            # 'mts_presequences': r'mitochondria-targeting sequence \(mts\) in the presequences.*?\n',
            # 'recognition_msf': r'recognition of mts of preproteins by cytosolic.*?\n',
            # 'interaction_outer': r'interaction of mts with the receptors.*?\n',
            # 'role_inner_membrane': r'role of mts in protein translocation across the inner membrane.*?\n',
            # 'necessity_processing_mpp': r'necessity of mts for the processing of imported.*?\n',
            # 'multi_step_recognition': r'recognition of mts at multiple steps during.*?\n'
        }

        sections = {}
        
        for name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.end()
                next_section = re.search(r'\n[A-Z ]+\n', text[start:], re.IGNORECASE)
                end = start + (next_section.start() if next_section else len(text))
                sections[name] = text[start:end].strip()

        

        unmatched_text = text
        for content in sections.values():
            unmatched_text = unmatched_text.replace(content, "")  # Remove already identified sections

        generic_matches = re.finditer(r'\n[A-Z][A-Za-z0-9\-: ]{3,}\n', unmatched_text)
        for match in generic_matches:
            start = match.end()
            next_section = re.search(r'\n[A-Z][A-Za-z0-9\-: ]{3,}\n', unmatched_text[start:])
            end = start + (next_section.start() if next_section else len(unmatched_text))
            header = match.group().strip()
            sections[header] = unmatched_text[start:end].strip()

        return sections
            


    def get_embeddings(self, text_list):
        embeddings = []
        print(len(text_list))
        for text in text_list:
            print(f"Processing text: {text[:50]}...")  # Print first 50 characters of each text
            tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            print(f"Tokenized input: {tokens}")

            with torch.no_grad():
                outputs = self.model(**tokens)
            # Check output shape
            print(f"Output shape: {outputs.last_hidden_state.shape}")
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
        print(f"Generated {len(embeddings)} embeddings.")
        return embeddings

    def cluster_texts(self, texts, n_clusters):
        """
        Performs KMeans clustering on a list of texts.
        """
        # Generate embeddings
        embeddings = self.get_embeddings(texts)

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        return clusters, embeddings

    def visualize_clusters(self, embeddings, clusters):
        """
        Visualizes clusters using PCA.
        """
        # Reduce dimensions with PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Plot
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        plt.title("KMeans Clustering")
        plt.show()

    def cluster_and_visualize(self, texts, n_clusters):
        """
        High-level method to cluster texts and visualize the results.
        """
        clusters, embeddings = self.cluster_texts(texts, n_clusters)
        self.visualize_clusters(embeddings, clusters)

        # Print results
        for i, text in enumerate(texts):
            print(f"Text: {text} -> Cluster: {clusters[i]}")


class DocumentClusterer:
    def __init__(self, paper_processor):
        self.processor = paper_processor

    def process_documents(self, pdf_paths):
        """
        Extracts text and generates embeddings for a list of PDF paths.
        """
        document_texts = []
        for pdf_path in pdf_paths:
            print(f"Processing: {pdf_path}")
            text = self.processor.extract_text(pdf_path)
            document_texts.append(text)
        return document_texts

    def generate_document_embeddings(self, document_texts):
        """
        Generates embeddings for entire documents.
        """
        embeddings = self.processor.get_embeddings(document_texts)
        return embeddings

    def cluster_documents(self, embeddings, n_clusters):
        """
        Clusters document embeddings using KMeans.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters

    def visualize_clusters(self, embeddings, clusters):
        """
        Visualizes document clusters using PCA.
        """
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        plt.title("Document Clustering")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

# 2. Content Analysis
class ContentAnalyzer:
    def __init__(self):
        self.bert_model = AutoModel.from_pretrained('bert-base-scibert')
        
    def extract_technical_terms(self, text):
        # Use NER and custom rules for technical term identification
        pass
        
    def analyze_citations(self, text):
        # Extract and analyze citation patterns
        citation_pattern = r'\[(.*?)\]'
        citations = re.findall(citation_pattern, text)
        return citations

# 3. Knowledge Graph Creation
import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_paper(self, paper_id, title, abstract):
        self.graph.add_node(paper_id, 
                           title=title, 
                           abstract=abstract)
        
    def add_citation(self, source_id, target_id):
        self.graph.add_edge(source_id, target_id)
        
    def analyze_centrality(self):
        return nx.pagerank(self.graph)

# 4. Study Aid Generation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class StudyAidGenerator:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    def generate_flashcards(self, text):
        # Extract key concepts and definitions
        terms = self.extract_technical_terms(text)
        flashcards = []
        for term in terms:
            definition = self.find_definition(term, text)
            flashcards.append({"term": term, "definition": definition})
        return flashcards

class MathHandler:
    def __init__(self):
        self.latex_patterns = {
            'inline': r'\$(.*?)\$',
            'display': r'\$\$(.*?)\$\$',
            'equation': r'\\begin{equation}(.*?)\\end{equation}'
        }
        
    def extract_formulas(self, text):
        formulas = []
        for pattern_type, pattern in self.latex_patterns.items():
            matches = re.findall(pattern, text, re.DOTALL)
            formulas.extend([{
                'type': pattern_type,
                'content': match,
                'simplified': self.simplify_formula(match)
            } for match in matches])
        return formulas
    
    def simplify_formula(self, formula):
        """Convert complex LaTeX to simplified form for ML processing"""
        # Remove formatting commands
        simplified = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', formula)
        # Convert subscripts/superscripts
        simplified = re.sub(r'_([^\s{])|_{([^}]*)}', r'_\1\2', simplified)
        return simplified

class IntegratedPaperAnalyzer:
    def __init__(self):
        # Initialize components
        self.preprocessor = PaperPreprocessor()
        self.math_handler = MathHandler()
        self.bert_model = AutoModel.from_pretrained('allenai/scibert')
        self.classifier = torch.nn.Linear(768, num_classes)  # BERT output size
        
    def analyze_paper(self, pdf_path):
        # 1. Extract and preprocess text
        text = self.extract_text(pdf_path)
        processed = self.preprocessor.process_paper(text)
        
        # 2. Process mathematical content
        math_content = self.math_handler.extract_formulas(text)
        
        # 3. Generate embeddings
        embeddings = self.generate_embeddings(processed['clean_text'])
        
        # 4. Classify sections and topics
        classifications = self.classify_content(embeddings)
        
        # 5. Create knowledge graph
        graph = self.build_knowledge_graph(processed, embeddings)
        
        return {
            'processed_text': processed,
            'math_content': math_content,
            'embeddings': embeddings,
            'classifications': classifications,
            'knowledge_graph': graph
        }
    
    def generate_embeddings(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**tokens)
        return outputs.last_hidden_state

class SmartSummarizer:
    def __init__(self):
        # Initialize a summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def generate_summary(self, text, max_length=400, min_length=30):
        """
        Generates a smart summary for the given text.
        """
        try:
            if len(text.split()) > 1024:
                print("Text is too long. Splitting into chunks.")

                chunks = self.split_text_into_chunks(text, chunk_size=1024)
                summaries = []
                
                for chunk in chunks:
                    chunk_text = " ".join(chunk)
                    summary = self.summarizer(
                        chunk_text, max_length=max_length, min_length=min_length, truncation=True
                    )
                    summaries.append(summary[0]['summary_text'])
                
                return " ".join(summaries)

        except Exception as e:
            print(f"Failed to generate summary: {e}")
            return "Summary could not be generated."
        
        # If the text is small enough, summarize directly
        summary = self.summarizer(
            text, max_length=max_length, min_length=min_length, truncation=True
        )
        return summary[0]['summary_text']

    def split_text_into_chunks(self, text, chunk_size=1024):
        """
        Splits the text into chunks of a specific token size.
        """
        tokens = self.summarizer.tokenizer.tokenize(text)
        token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return [self.summarizer.tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]