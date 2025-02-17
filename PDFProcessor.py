import fitz  # extract text
import torch #deep 
import re
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
from sklearn.cluster import KMeans #clustering data 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import requests
import pickle

#Due to 403 error request, we cannot mass download PDFs. We can however mass extract abstracts. Hence, two functionalities aimed are
#Large knowledge graph with abstracts
#Moderate knowledge graph with manually downloaded documents which are then summarized

class PaperProcessor:
    def __init__(self, knowledge_graph_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter', timeout=60) 
        self.model = AutoModel.from_pretrained('allenai/specter')
        self.summarizer = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.knowledge_graph = KnowledgeGraph()
        self.document_clusterer = DocumentClusterer()

        if knowledge_graph_path:
            self.knowledge_graph = self.load_knowledge_graph(knowledge_graph_path)
        else:
            self.knowledge_graph = KnowledgeGraph()

        

    #self explanatoyr
    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text() for page in doc])

    # def extract_bold_headers(pdf_path):
    #     doc = fitz.open(pdf_path)
    #     bold_headers = []

    #     for page_num in range(len(doc)):
    #         page = doc.load_page(page_num)
    #         text_instances = page.extract_text("dict")  # Extract text in dictionary format
            
    #         # Loop through text instances
    #         for block in text_instances["blocks"]:
    #             for line in block["lines"]:
    #                 for span in line["spans"]:
    #                     # Check if the span's font is bold
    #                     if "bold" in span["font"].lower():  # Check for the word "bold" in the font name
    #                         bold_headers.append(span["text"])

    #     return bold_headers

    def process_paper(self, paper_id, pdf_path):
        # Extract text and sections from the paper
        text = self.extract_text(pdf_path)
        sections = self.identify_sections(text)
        abstract = sections.get("abstract", "")
        title = sections.get("title", "Unknown Title")

        # Add the paper to the knowledge graph
        self.knowledge_graph.add_paper(paper_id, title, abstract)

        # Generate and print summary
        summary = self.summarize_text(text)
        self.document_clusterer.add_summary(summary)
        print(f"Summary for {title}: {summary}")

    
    def summarize_text(self, text, max_length=50):
        """Generates a summary of the text with a max_length limit."""
        inputs = self.summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = self.summarizer.generate(inputs, max_length=max_length, min_length=10, length_penalty=2.0)
        return self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    def successive_summarization(self, text, chunk_size=500, summary_length=50, overlap=0):
        words = text.split()
    
    # Create overlapping chunks
        chunks = [
            " ".join(words[i:min(i+chunk_size, len(words))])
            for i in range(0, len(words), chunk_size - overlap)
        ]
        
        # First-level summaries
        summaries = [self.summarize_text(chunk, summary_length) for chunk in chunks]

        # If the combined summary is still long, summarize again
        final_summary = " ".join(summaries)
        # if len(final_summary.split()) > summary_length * 2:
        #     final_summary = self.summarize_text(final_summary, summary_length)

        return final_summary

    def save_knowledge_graph(self, path):
        """Saves the knowledge graph to a file."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.knowledge_graph.graph, f)
            print(f"Knowledge graph saved to {path}")
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")

    def load_knowledge_graph(self, path):
        """Loads the knowledge graph from a file."""
        try:
            with open(path, 'rb') as f:
                graph = pickle.load(f)
                self.knowledge_graph.graph = graph
            print(f"Knowledge graph loaded from {path}")
            return 
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            return None
        

    
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        
    def add_paper(self, paper_id, title, abstract):
        self.graph.add_node(paper_id, 
                           title=title, 
                           abstract=abstract)
        
    def add_edge(self, source_id, target_id):
        self.graph.add_edge(source_id, target_id)
        
    def analyze_centrality(self):
        return nx.pagerank(self.graph)

    def get_paper_info(self, paper_id):
        """Retrieves the information (title, abstract) of a paper."""
        return self.graph.nodes[paper_id] if paper_id in self.graph else None

    def compute_text_similarity(self, threshold=0.3):
        """Computes cosine similarity between paper abstracts and adds edges based on a threshold."""
        abstracts = {node: data["abstract"] for node, data in self.graph.nodes(data=True)}

        if len(abstracts) < 2:
            print("Not enough papers to compute similarity.")
            return

        paper_ids, texts = zip(*abstracts.items())

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        similarities = cosine_similarity(tfidf_matrix)

        for i in range(len(paper_ids)):
            for j in range(i + 1, len(paper_ids)):
                if similarities[i, j] > threshold:
                    self.graph.add_edge(paper_ids[i], paper_ids[j], weight=similarities[i, j])

        print("Edges added based on text similarity.")

    def visualize(self):
        """Visualizes the graph using NetworkX and Matplotlib."""
        print("Visualizing Knowledge Graph...")  # Debugging line
        if not self.graph.nodes:
            print("Graph is empty, nothing to visualize!")
            return
        else:
            print("This graph has something")
        pos = nx.spring_layout(self.graph, seed=40, k=0.8)
        plt.figure(figsize=(50, 50))
        nx.draw(self.graph, pos, with_labels=True, node_size=4000, node_color='skyblue', font_size=5, edge_color='gray')
        plt.show()

    def summarize_graph(self):
        """Summarizes the graph by returning the papers and their citation relationships."""
        summary = {}
        for node in self.graph.nodes:
            summary[node] = {
                'title': self.graph.nodes[node].get('title'),
                'abstract': self.graph.nodes[node].get('abstract'),
                'citations': list(self.graph.neighbors(node))
            }
        return summary
    
class PaperFetcher:
    def __init__(self, api_key=None):
        #self.api_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": api_key} if api_key else {}
    
    def search_papers(self, query, limit=20):
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,paperId,url,openAccessPdf"
        }
        response = requests.get(base_url, params=params)
        print("Status Code:", response.status_code)
        try:
            data = response.json()
            #print("Response Data:", data)  # Print full response to debug
            papers = data.get("data", [])
            #print(papers)
            #open_access_papers = [paper for paper in papers if paper.get("openAccessPdf")]
            return papers
        except Exception as e:
            print("Error parsing JSON:", e)
            return []


        return data.get("data", [])

    def get_paper_details(self, paper_id):
        base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {"fields": "title,abstract,authors,year,citationCount,url,openAccessPdf"}
        response = requests.get(base_url, params=params)
        print(response.json())
        return response.json()


class DocumentClusterer:
    def __init__(self, model_name='distilbert-base-nli-stsb-mean-tokens'):
        # Initialize the embedding model (Sentence-BERT)
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        """
        Generates embeddings for the input texts using a transformer model.
        """
        embeddings = self.model.encode(texts)
        return embeddings

    def cluster_texts(self, texts, n_clusters):
        """
        Performs KMeans clustering on a list of texts.
        """
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Scale the embeddings (important for KMeans)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings_scaled)
        
        return clusters, embeddings, kmeans

    def visualize_clusters(self, embeddings, clusters, kmeans):
        """
        Visualizes clusters using PCA.
        """
        # Reduce dimensions with PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        
        # Plot cluster centers
        centers = kmeans.cluster_centers_
        reduced_centers = pca.transform(centers)
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], marker='x', s=200, c='red')

        plt.title("KMeans Clustering with PCA")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Cluster ID")
        plt.show()

    def cluster_and_visualize(self, texts, n_clusters):
        """
        High-level method to cluster texts and visualize the results.
        """
        clusters, embeddings, kmeans = self.cluster_texts(texts, n_clusters)
        self.visualize_clusters(embeddings, clusters, kmeans)

        # Print results
        for i, text in enumerate(texts):
            print(f"Text: {text[:100]}... -> Cluster: {clusters[i]}")  # Only show first 100 characters for brevity









            
    # def get_embeddings(self, text_list):
    #     embeddings = []
    #     print(len(text_list))
    #     for text in text_list:
    #         print(f"Processing text: {text[:50]}...")  # Print first 50 characters of each text
    #         tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    #         print(f"Tokenized input: {tokens}")

    #         with torch.no_grad():
    #             outputs = self.model(**tokens)
    #         # Check output shape
    #         print(f"Output shape: {outputs.last_hidden_state.shape}")
    #         embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    #     print(f"Generated {len(embeddings)} embeddings.")
    #     return embeddings

        # def cluster_texts(self, texts, n_clusters):
        #     """
        #     Performs KMeans clustering on a list of texts.
        #     """
        #     # Generate embeddings
        #     embeddings = self.get_embeddings(texts)

        #     # Apply KMeans
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        #     clusters = kmeans.fit_predict(embeddings)

        #     return clusters, embeddings

        # def visualize_clusters(self, embeddings, clusters):
        #     """
        #     Visualizes clusters using PCA.
        #     """
        #     # Reduce dimensions with PCA
        #     pca = PCA(n_components=2)
        #     reduced_embeddings = pca.fit_transform(embeddings)

        #     # Plot
        #     plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        #     plt.title("KMeans Clustering")
        #     plt.show()


        # def cluster_and_visualize(self, texts, n_clusters):
        #     """
        #     High-level method to cluster texts and visualize the results.
        #     """
        #     clusters, embeddings = self.cluster_texts(texts, n_clusters)
        #     self.visualize_clusters(embeddings, clusters)

        #     # Print results
        #     for i, text in enumerate(texts):
        #         print(f"Text: {text} -> Cluster: {clusters[i]}")
#

# # 2. Content Analysis
# class ContentAnalyzer:
#     def __init__(self):
#         self.bert_model = AutoModel.from_pretrained('bert-base-scibert')
        
#     def extract_technical_terms(self, text):
#         # Use NER and custom rules for technical term identification
#         pass
        
#     def analyze_citations(self, text):
#         # Extract and analyze citation patterns
#         citation_pattern = r'\[(.*?)\]'
#         citations = re.findall(citation_pattern, text)
#         return citations

# # 3. Knowledge Graph Creation
# import networkx as nx



# 4. Study Aid Generation
# from transformers import GPT2LMHeadModel, GPT2Tokenizer


# class SmartSummarizer:
#     def __init__(self):
#         # Initialize a summarization pipeline
#         self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#     def generate_summary(self, text, max_length=400, min_length=30):
#         """
#         Generates a smart summary for the given text.
#         """
#         try:
#             if len(text.split()) > 1024:
#                 print("Text is too long. Splitting into chunks.")

#                 chunks = self.split_text_into_chunks(text, chunk_size=1024)
#                 summaries = []
                
#                 for chunk in chunks:
#                     chunk_text = " ".join(chunk)
#                     summary = self.summarizer(
#                         chunk_text, max_length=max_length, min_length=min_length, truncation=True
#                     )
#                     summaries.append(summary[0]['summary_text'])
                
#                 return " ".join(summaries)

#         except Exception as e:
#             print(f"Failed to generate summary: {e}")
#             return "Summary could not be generated."
        
#         # If the text is small enough, summarize directly
#         summary = self.summarizer(
#             text, max_length=max_length, min_length=min_length, truncation=True
#         )
#         return summary[0]['summary_text']

#     def split_text_into_chunks(self, text, chunk_size=1024):
#         """
#         Splits the text into chunks of a specific token size.
#         """
#         tokens = self.summarizer.tokenizer.tokenize(text)
#         token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
#         return [self.summarizer.tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]

        
# class StudyAidGenerator:
#     def __init__(self):
#         self.model = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
#     def generate_flashcards(self, text):
#         # Extract key concepts and definitions
#         terms = self.extract_technical_terms(text)
#         flashcards = []
#         for term in terms:
#             definition = self.find_definition(term, text)
#             flashcards.append({"term": term, "definition": definition})
#         return flashcards

# class MathHandler:
#     def __init__(self):
#         self.latex_patterns = {
#             'inline': r'\$(.*?)\$',
#             'display': r'\$\$(.*?)\$\$',
#             'equation': r'\\begin{equation}(.*?)\\end{equation}'
#         }
        
#     def extract_formulas(self, text):
#         formulas = []
#         for pattern_type, pattern in self.latex_patterns.items():
#             matches = re.findall(pattern, text, re.DOTALL)
#             formulas.extend([{
#                 'type': pattern_type,
#                 'content': match,
#                 'simplified': self.simplify_formula(match)
#             } for match in matches])
#         return formulas
    
#     def simplify_formula(self, formula):
#         """Convert complex LaTeX to simplified form for ML processing"""
#         # Remove formatting commands
#         simplified = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', formula)
#         # Convert subscripts/superscripts
#         simplified = re.sub(r'_([^\s{])|_{([^}]*)}', r'_\1\2', simplified)
#         return simplified

# class IntegratedPaperAnalyzer:
#     def __init__(self):
#         # Initialize components
#         self.preprocessor = PaperPreprocessor()
#         self.math_handler = MathHandler()
#         self.bert_model = AutoModel.from_pretrained('allenai/scibert')
#         self.classifier = torch.nn.Linear(768, num_classes)  # BERT output size
        
#     def analyze_paper(self, pdf_path):
#         # 1. Extract and preprocess text
#         text = self.extract_text(pdf_path)
#         processed = self.preprocessor.process_paper(text)
        
#         # 2. Process mathematical content
#         math_content = self.math_handler.extract_formulas(text)
        
#         # 3. Generate embeddings
#         embeddings = self.generate_embeddings(processed['clean_text'])
        
#         # 4. Classify sections and topics
#         classifications = self.classify_content(embeddings)
        
#         # 5. Create knowledge graph
#         graph = self.build_knowledge_graph(processed, embeddings)
        
#         return {
#             'processed_text': processed,
#             'math_content': math_content,
#             'embeddings': embeddings,
#             'classifications': classifications,
#             'knowledge_graph': graph
#         }
    
#     def generate_embeddings(self, text):
#         tokens = self.tokenizer(text, return_tensors='pt', padding=True)
#         with torch.no_grad():
#             outputs = self.bert_model(**tokens)
#         return outputs.last_hidden_state