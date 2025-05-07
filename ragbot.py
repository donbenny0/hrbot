import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Updated import as per deprecation warning
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
file_path = "docs/story.txt"

def readDocs():
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def create_initial_chunks():
    text = readDocs()
    # Split text by periods but keep the periods with the preceding text
    paragraphs = [p + '.' for p in re.split(r'\.', text)[:-1]]
    # Add the last chunk without adding an extra period
    if text and not text.endswith('.'):
        paragraphs.append(re.split(r'\.', text)[-1])
    return paragraphs

def create_semantic_chunks():
    paragraphs = create_initial_chunks()
    # Fixed this line to embed each paragraph individually
    para_embeddings = [np.array(embeddings.embed_query(paragraph)).reshape(1,-1) for paragraph in paragraphs]
    
    semantic_chunks = []
    for i in range(len(paragraphs)):
        if i == 0:
            semantic_chunks.append([paragraphs[i]])
        else:
            similarity = cosine_similarity(para_embeddings[i-1], para_embeddings[i])
            if similarity[0][0] > 0.5:
                semantic_chunks[-1].append(paragraphs[i])
            else:
                semantic_chunks.append([paragraphs[i]])
    
    return semantic_chunks

# Print chunks in a verifiable format
chunks = create_semantic_chunks()
print(f"Total semantic chunks: {len(chunks)}\n")

for i, chunk_group in enumerate(chunks, 1):
    print(f"Semantic Chunk Group {i} (containing {len(chunk_group)} segments):")
    for j, segment in enumerate(chunk_group, 1):
        print(f"  {j}. {segment.strip()}")
    print("-" * 60)  # Separator between chunk groups