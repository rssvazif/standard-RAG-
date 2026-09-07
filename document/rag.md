# Retrieval-Augmented Generation

## What is RAG?

Retrieval-Augmented Generation, or RAG, is an architecture that combines
information retrieval with a generative language model.

Instead of asking the language model to answer only from its parameters,
the application first retrieves relevant information from an external
knowledge base and provides that information to the model as context.

## Ingestion Pipeline

The ingestion pipeline prepares documents before users ask questions.

A typical pipeline contains document parsing, normalization, chunking,
metadata enrichment, embedding generation, and storage in a vector database.

### Document Parsing

Document parsing converts a raw document into a representation that the
application can process.

For Markdown, parsing can identify headings, sections, paragraphs, and
other structural elements.

### Sliding Window Chunking

Sliding window chunking divides text into fixed-size chunks while keeping
an overlap between neighboring chunks.

For example, with a chunk size of 500 characters and an overlap of
100 characters, the next chunk starts 400 characters after the previous
chunk started.

The overlap helps preserve context when an important sentence is located
near a chunk boundary.

### Metadata Enrichment

Metadata provides additional information about each chunk.

Useful metadata can include the source file, section or heading path,
file type, chunk index, repository, document version, or access-control
information.

Metadata can later be used for filtering, tracing, debugging, and
explaining where retrieved information came from.

## Retrieval

During retrieval, the user query is transformed into an embedding and
compared with document embeddings in the vector database.

Dense retrieval is useful for semantic similarity.

Sparse retrieval, such as BM25, is useful for lexical matching and exact
terms such as identifiers, error codes, class names, and API names.

### Hybrid Retrieval

Hybrid retrieval combines dense and sparse retrieval.

Dense retrieval captures semantic similarity, while sparse retrieval
captures lexical similarity. Combining both signals can improve recall
across different types of queries.

### Fusion

The results from different retrievers need to be combined into one
ranking.

Reciprocal Rank Fusion, or RRF, is one common approach. RRF combines
rank positions instead of directly adding scores from different retrieval
systems.

## Generation

After retrieval and fusion, the most relevant chunks are placed into the
LLM context.

The language model uses this retrieved context to generate an answer.
A good RAG system should instruct the model to rely on the provided
knowledge base and state when the required information is not available.