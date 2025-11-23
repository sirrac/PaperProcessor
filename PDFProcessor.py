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
import arxiv

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

    def fetch_from_arxiv(self, query):
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in search.results():
            print(f"Downloading {result.title}...")
            result.download_pdf(dirpath="./SamplePapers", filename=f"{result.entry_id.split('/')[-1]}.pdf")


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








