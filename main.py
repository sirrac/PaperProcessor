import os
import requests
import time
import random
from PDFProcessor import PaperFetcher
from PDFProcessor import PaperProcessor
from PDFProcessor import DocumentClusterer
from PDFProcessor import KnowledgeGraph

def clusterTest():
    # Sample text data (list of documents)
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Artificial intelligence includes subfields like machine learning and deep learning.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language for data science.",
        "Data science involves the use of statistical methods and machine learning techniques."
    ]
    
    # Create an instance of DocumentClusterer
    clusterer = DocumentClusterer()

    # Perform clustering and visualization
    n_clusters = 2  # Set the number of clusters
    clusterer.cluster_and_visualize(texts, n_clusters)


clusterTest()
processor = PaperProcessor()
# # Create save directory
# save_dir = r"C:\Users\srika\.vscode\PaperProcessor\SamplePapers"
# os.makedirs(save_dir, exist_ok=True)

# # List of paper paths
fetcher = PaperFetcher()

fetcher.fetch_from_arxiv("quantum physics")

# pdf_folder = r"C:\Users\srika\.vscode\PaperProcessor\SamplePapers"
# pdf_paths = [os.path.join(pdf_folder, filename) for filename in os.listdir(pdf_folder) if filename.endswith('.pdf')]

# print(pdf_paths)

# # Process each PDF
# for idx, pdf_path in enumerate(pdf_paths):
#     print(f"Processing {pdf_path}...")
#     text = processor.extract_text(pdf_path)
#     summarized_text = processor.successive_summarization(text)
    
#     # Use the filename as a unique paper ID
#     paper_id = pdf_path.split("\\")[-1]  
#     processor.knowledge_graph.add_paper(paper_id, paper_id, summarized_text)

# print("Nodes in the Knowledge Graph:", processor.knowledge_graph.graph.nodes(data=True))

# # Analyze centrality of the graph
# centrality_scores = processor.knowledge_graph.analyze_centrality()
# processor.knowledge_graph.compute_text_similarity(threshold=0.3)

# print("\nPaper Centrality Scores:")
# for paper, score in centrality_scores.items():
#     print(f"{paper}: {score:.4f}")

# print("\nProcessing complete.")

# processor.knowledge_graph.visualize()

# processor.save_knowledge_graph(r"C:\Users\srika\.vscode\PaperProcessor\knowledge_graph.pkl")


# new_processor = PaperProcessor()

# # Load the saved knowledge graph
# new_processor.load_knowledge_graph(r"C:\Users\srika\.vscode\PaperProcessor\knowledge_graph.pkl")

# # Check if the nodes loaded correctly
# # print("Loaded Knowledge Graph Nodes:", new_processor.knowledge_graph.graph.nodes(data=True))

# # Re-run any functions on the loaded graph to verify it works
# new_processor.knowledge_graph.visualize()    


# # Search query
# query = "mathematics"
# papers = fetcher.search_papers(query, limit=20)

# #
# print("Search Results:")
# for paper in papers:
#     print(f"Title: {paper.get('title')}")
#     #print(f"Authors: {[author['name'] for author in paper.get('authors', [])]}")
#     print(f"paperId: {paper.get('paperId')}")
#     print(f"URL: {paper.get('url')}")
#     print(f"Open access {paper.get('openAccessPdf')}")
#     print("-" * 50)

# if papers:
#     #first_paper_id = papers[0].get("paperId")

#     for paper in papers:
#         paper_id = paper.get("paperId")

#         paper_details = fetcher.get_paper_details(paper_id)

#         if paper_details:

#             print("\nPaper Details:")
#             print(f"Title: {paper_details.get('title')}")
#             print(f"Abstract: {paper_details.get('abstract')}")
#             print(f"Authors: {[author['name'] for author in paper_details.get('authors', [])]}")
#             print(f"Year: {paper_details.get('year')}")
#             print(f"Citations: {paper_details.get('citationCount')}")      
#             print(f"URL: {paper_details.get('url')}")
#             print(f"Open Access: {paper_details.get('openAccessPdf')}")
#         else:
#             print("No details found!")

#Manually downloaded process not viable due to 403 errors

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
#     "Accept-Encoding": "gzip, deflate, br",
#     "Accept-Language": "en-US,en;q=0.9",
#     "Connection": "keep-alive",
#     "Upgrade-Insecure-Requests": "1"
# }

# # Function to download paper with retry and random delay
# def download_paper(pdf_url, pdf_path):
#     try:
#         # Request the paper with headers
#         response = requests.get(pdf_url, headers=headers, stream=True)
        
#         # Check for status codes
#         if response.status_code == 429:
#             raise requests.exceptions.TooManyRedirects("Too many requests - rate limited.")
        
#         response.raise_for_status()  # Raise error if request fails

#         # Save the file
#         with open(pdf_path, "wb") as pdf_file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 pdf_file.write(chunk)

#         print(f"Downloaded: {pdf_path}")

#     except requests.exceptions.TooManyRedirects:
#         print(f"Rate limit hit for {pdf_url}. Retrying...")
#         return False
#     except requests.exceptions.RequestException as e:
#         print(f"Failed to download {pdf_url}: {e}")
#         return False

#     return True

# # Download PDFs with retry mechanism and exponential backoff
# for i, paper in enumerate(papers):
#     pdf_info = paper.get("openAccessPdf")
    
#     if pdf_info:
#         pdf_url = pdf_info.get("url")
#         title = paper.get("title", f"paper_{i+1}").replace(" ", "_").replace("/", "_")  # Clean filename
        
#         # Define the save path
#         pdf_path = os.path.join(save_dir, f"{title}.pdf")
        
#         # Retry mechanism with backoff strategy
#         retries = 5
#         backoff_time = 1  # Start with 1 second

#         for attempt in range(retries):
#             success = download_paper(pdf_url, pdf_path)
#             if success:
#                 break
#             else:
#                 print(f"Retrying... {attempt + 1}/{retries}")
#                 time.sleep(backoff_time)  # Exponential backoff
#                 backoff_time *= 2  # Double the delay after each retry

#         # Delay between each download to avoid hitting rate limits
#         time.sleep(random.uniform(3, 5))  # Increased delay between downloads

# print("Download process completed!")

# print("Download process completed!")

