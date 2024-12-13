#import PaperPreprocessor
from PDFProcessor import PaperProcessor
from PDFProcessor import SmartSummarizer
#from visualize import PaperVisualizer

# processor = PaperProcessor()

# pdf_path = r"C:\Users\srika\.vscode\PaperProcessor\SamplePapers\Mitochondria-Target.pdf"  
# print("Extracting text from the PDF...")
# text = processor.extract_text(pdf_path)
# print(f"Extracted text (first 500 characters):\n{text[:500]}")

# print("\nIdentifying sections...")
# sections = processor.identify_sections(text)
# for section, content in sections.items():
#     print(f"\n--- {section.upper()} ---\n{content[:500]}")  # Display first 500 characters of each section

# # Step 4: Cluster sections
# print("\nClustering sections...")
# section_texts = list(sections.values())
# section_clusters, embeddings = processor.cluster_texts(section_texts, n_clusters=3)
# print("\nCluster Assignments:")
# for section, cluster in zip(sections.keys(), section_clusters):
#     print(f"Section: {section} -> Cluster: {cluster}")

# # Step 5: Visualize clusters
# print("\nVisualizing clusters...")
# processor.visualize_clusters(embeddings, section_clusters)

processor = PaperProcessor()
summarizer = SmartSummarizer()

# Path to a PDF
pdf_path = r"C:\Users\srika\.vscode\PaperProcessor\SamplePapers\Mitochondria-Target.pdf"  

# Step 1: Extract text
print("Extracting text...")
text = processor.extract_text(pdf_path)
print(f"Extracted text length: {len(text)}")

# Step 2: Generate summary
print("Generating summary...")
summary = summarizer.generate_summary(text)
print(f"\nSmart Summary:\n{summary}")