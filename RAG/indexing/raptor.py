# Standard library imports
import os
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import tiktoken
import umap
from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture

# LangChain imports
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Calculate the number of tokens in a text string.
    
    Args:
        string: The text string to count tokens for.
        encoding_name: The name of the encoding to use (e.g., 'cl100k_base').
        
    Returns:
        The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Perform global dimensionality reduction on embeddings using UMAP.

    Args:
        embeddings: The input embeddings as a numpy array.
        dim: The target dimensionality for the reduced space.
        n_neighbors: The number of neighbors to consider for each point.
                    If not provided, defaults to sqrt of number of embeddings.
        metric: The distance metric to use for UMAP.

    Returns:
        A numpy array of embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """Perform local dimensionality reduction using UMAP after global clustering.

    Args:
        embeddings: The input embeddings as a numpy array.
        dim: The target dimensionality for the reduced space.
        num_neighbors: The number of neighbors to consider for each point.
        metric: The distance metric to use for UMAP.

    Returns:
        A numpy array of embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 42
) -> int:
    """Determine optimal number of clusters using BIC with Gaussian Mixture Model.

    Args:
        embeddings: The input embeddings as a numpy array.
        max_clusters: The maximum number of clusters to consider.
        random_state: Seed for reproducibility.

    Returns:
        An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def gmm_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """Cluster embeddings using Gaussian Mixture Model based on probability threshold.

    Args:
        embeddings: The input embeddings as a numpy array.
        threshold: The probability threshold for assigning an embedding to a cluster.
        random_state: Seed for reproducibility.

    Returns:
        A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """Perform clustering using global and local dimensionality reduction.

    This function first reduces dimensionality globally, then clusters using GMM,
    and finally performs local clustering within each global cluster.

    Args:
        embeddings: The input embeddings as a numpy array.
        dim: The target dimensionality for UMAP reduction.
        threshold: The probability threshold for assigning embeddings to clusters.

    Returns:
        A list of numpy arrays, where each array contains cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = gmm_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = gmm_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def embed_texts(texts: List[str], embeddings_model: OpenAIEmbeddings) -> np.ndarray:
    """Generate embeddings for a list of text documents.

    Args:
        texts: A list of text documents to be embedded.
        embeddings_model: The OpenAI embeddings model to use.

    Returns:
        A numpy array of embeddings for the given text documents.
    """
    text_embeddings = embeddings_model.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_cluster_texts(texts: List[str], embeddings_model: OpenAIEmbeddings) -> pd.DataFrame:
    """Embed and cluster a list of texts, returning a DataFrame with results.

    Args:
        texts: A list of text documents to be processed.
        embeddings_model: The OpenAI embeddings model to use.

    Returns:
        A DataFrame containing the original texts, their embeddings, and cluster labels.
    """
    text_embeddings_np = embed_texts(texts, embeddings_model)
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)
    
    df = pd.DataFrame()
    df["text"] = texts
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels
    return df


def format_cluster_texts(df: pd.DataFrame) -> str:
    """Format text documents in a DataFrame into a single string.

    Args:
        df: DataFrame containing the 'text' column with text documents to format.

    Returns:
        A single string where all text documents are joined by a delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts(
    texts: List[str], level: int, embeddings_model: OpenAIEmbeddings, llm_model: ChatOpenAI
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Embed, cluster, and summarize a list of texts.

    This function generates embeddings, clusters them based on similarity,
    expands cluster assignments, and summarizes content within each cluster.

    Args:
        texts: A list of text documents to be processed.
        level: An integer parameter defining the depth or detail of processing.
        embeddings_model: The OpenAI embeddings model to use.
        llm_model: The ChatOpenAI model to use for summarization.

    Returns:
        A tuple containing two DataFrames:
        1. df_clusters: includes original texts, embeddings, and cluster assignments
        2. df_summary: contains summaries for each cluster with level and cluster identifiers
    """
    # Embed and cluster the texts
    df_clusters = embed_cluster_texts(texts, embeddings_model)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")

    # Summarization template
    template = """Here is a sub-set of LangChain Expression Language doc. 
    
    LangChain Expression Language provides a way to compose chain in LangChain.
    
    Give a detailed summary of the documentation provided.
    
    Documentation:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm_model | StrOutputParser()

    # Format text within each cluster for summarization
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = format_cluster_texts(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    # Create a DataFrame to store summaries
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], 
    level: int = 1, 
    n_levels: int = 3,
    embeddings_model: OpenAIEmbeddings = None,
    llm_model: ChatOpenAI = None
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Recursively embed, cluster, and summarize texts up to a specified level.

    Args:
        texts: List of texts to be processed.
        level: Current recursion level (starts at 1).
        n_levels: Maximum depth of recursion.
        embeddings_model: The OpenAI embeddings model to use.
        llm_model: The ChatOpenAI model to use for summarization.

    Returns:
        A dictionary where keys are recursion levels and values are tuples containing
        the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(
        texts, level, embeddings_model, llm_model
    )

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels, embeddings_model, llm_model
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results


def format_docs(docs):
    """Format documents for RAG chain input.
    
    Args:
        docs: List of documents to format.
        
    Returns:
        A formatted string of all document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def load_documents():
    """Load documents from various LangChain documentation sources.
    
    Returns:
        A list of loaded documents and their texts.
    """
    print("🌐 Loading documents from LangChain documentation...")
    
    # LCEL docs
    url = "https://python.langchain.com/docs/concepts/lcel/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()
    
    # LCEL w/ PydanticOutputParser
    url = "https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/quick_start/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs_pydantic = loader.load()
    
    # LCEL w/ Self Query
    url = "https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/self_query/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs_sq = loader.load()
    
    # Combine all documents
    docs.extend([*docs_pydantic, *docs_sq])
    docs_texts = [d.page_content for d in docs]
    
    return docs, docs_texts


def analyze_documents(docs_texts: List[str]):
    """Analyze token statistics of the loaded documents.
    
    Args:
        docs_texts: List of document texts to analyze.
    """
    print("\n📊 Analyzing document statistics...")
    
    # Calculate token counts
    counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]
    
    # Print statistics
    print(f"📄 Total number of documents: {len(docs_texts)}")
    print(f"🔢 Total tokens across all documents: {sum(counts):,}")
    print(f"📈 Average tokens per document: {sum(counts) / len(counts):.2f}")
    print(f"📉 Min tokens in a document: {min(counts):,}")
    print(f"📊 Max tokens in a document: {max(counts):,}")
    print(f"📏 Token count range: {max(counts) - min(counts):,}")


def build_raptor_tree(docs_texts: List[str], embeddings_model: OpenAIEmbeddings, llm_model: ChatOpenAI):
    """Build the RAPTOR tree structure by recursively clustering and summarizing.
    
    Args:
        docs_texts: List of document texts to process.
        embeddings_model: The OpenAI embeddings model to use.
        llm_model: The ChatOpenAI model to use for summarization.
        
    Returns:
        A list of all texts including original documents and summaries from all levels.
    """
    print("\n🌳 Building RAPTOR tree structure...")
    
    # Build tree through recursive clustering and summarization
    results = recursive_embed_cluster_summarize(
        docs_texts, level=1, n_levels=3, embeddings_model=embeddings_model, llm_model=llm_model
    )
    
    # Initialize all_texts with original documents
    all_texts = docs_texts.copy()
    
    # Extract summaries from each level and add to all_texts
    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)
        print(f"✅ Level {level}: Added {len(summaries)} summaries")
    
    print(f"🎯 Total texts in RAPTOR tree: {len(all_texts)}")
    return all_texts


def setup_rag_chain(all_texts: List[str], embeddings_model: OpenAIEmbeddings, llm_model: ChatOpenAI):
    """Set up the RAG chain with RAPTOR vectorstore.
    
    Args:
        all_texts: List of all texts including original documents and summaries.
        embeddings_model: The OpenAI embeddings model to use.
        llm_model: The ChatOpenAI model to use for generation.
        
    Returns:
        The configured RAG chain.
    """
    print("\n🔗 Setting up RAG chain...")
    
    # Create vectorstore with all texts
    vectorstore = Chroma.from_texts(texts=all_texts, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()
    
    # Set up RAG chain
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    
    print("✅ RAG chain ready!")
    return rag_chain


def main():
    """Main function to execute the RAPTOR RAG pipeline.
    
    This function demonstrates the complete RAPTOR (Recursive Abstractive Processing
    for Tree-Organized Retrieval) approach:
    1. Load documents from LangChain documentation
    2. Analyze document statistics
    3. Build RAPTOR tree through recursive clustering and summarization
    4. Set up RAG chain with hierarchical retrieval
    5. Test the system with a sample query
    """
    print("🚀 Starting RAPTOR RAG Pipeline")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    
    # Initialize models
    embeddings_model = OpenAIEmbeddings()
    llm_model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    # Load and analyze documents
    docs, docs_texts = load_documents()
    analyze_documents(docs_texts)
    
    # Build RAPTOR tree
    all_texts = build_raptor_tree(docs_texts, embeddings_model, llm_model)
    
    # Set up RAG chain
    rag_chain = setup_rag_chain(all_texts, embeddings_model, llm_model)
    
    # Test the system
    print("\n🎯 Testing RAPTOR RAG system...")
    question = "How to define a RAG chain? Give me a specific code example."
    print(f"❓ Question: {question}")
    
    result = rag_chain.invoke(question)
    print("\n💡 Answer:")
    print("-" * 50)
    print(result)
    print("-" * 50)
    
    print("\n🎉 RAPTOR RAG Pipeline completed successfully!")


if __name__ == "__main__":
    main()