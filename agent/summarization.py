"""
This code is a LangGraph implementation of a summarization pipeline.
It uses a map-reduce approach to summarize long texts via parallelization.
It also includes a conditional edge to collapse summaries if needed.
"""

import asyncio
import operator
from typing import Annotated, List, Optional, TypedDict
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

load_dotenv()


@dataclass
class SummarizationConfig:
    """
    Configuration class for the summarization pipeline.
    
    Attributes:
        model_name (str): LLM model name to use
        temperature (float): LLM temperature setting
        chunk_size (int): Chunk size for document splitting
        chunk_overlap (int): Overlap between chunks
        token_max (int): Maximum token count
        recursion_limit (int): Recursion limit for the pipeline
    """
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 0
    token_max: int = 1000
    recursion_limit: int = 10


class OverallState(TypedDict):
    """TypedDict for managing the overall summarization process state."""
    contents: List[str]
    summaries: Annotated[List[str], operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    """TypedDict for managing individual summary task state."""
    content: str


class DocumentSummarizer:
    """
    Main class for document summarization.
    
    Uses LangGraph to efficiently summarize long documents with Map-Reduce pattern.
    """
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize DocumentSummarizer.
        
        Args:
            config (Optional[SummarizationConfig]): Summarization config. Uses default if None.
        """
        self.config = config or SummarizationConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name, 
            temperature=self.config.temperature
        )
        
        # Setup prompts
        self._setup_prompts()
        
        # Setup text splitter
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # LangGraph app
        self.app = None
        
        print(f"✅ DocumentSummarizer initialized with model: {self.config.model_name}")
    
    def _setup_prompts(self) -> None:
        """Setup prompt templates."""
        # Stuff chain prompt
        self.stuff_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a concise summary of the following: \n\n{context}")
        ])
        
        # Map prompt (from hub)
        try:
            self.map_prompt = hub.pull("rlm/map-prompt")
        except Exception as e:
            print(f"⚠️  Failed to load map-prompt from hub: {e}. Using default prompt.")
            self.map_prompt = ChatPromptTemplate.from_messages([
                ("human", "Write a concise summary of the following:\n\n{context}")
            ])
        
        # Reduce prompt
        reduce_template = """
        The following is a set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary
        of the main themes.
        """
        self.reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    
    def load_documents(self, url: str) -> List[Document]:
        """
        Load and split documents from a web URL.
        
        Args:
            url (str): Web page URL to load
            
        Returns:
            List[Document]: List of split documents
            
        Raises:
            Exception: If document loading fails
        """
        try:
            print(f"📄 Loading documents from: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            split_docs = self.text_splitter.split_documents(docs)
            print(f"✂️  Document splitting completed: {len(split_docs)} chunks created")
            
            return split_docs
            
        except Exception as e:
            print(f"❌ Document loading failed: {e}")
            raise
    
    def simple_summarize(self, docs: List[Document]) -> str:
        """
        Summarize documents using simple Stuff method.
        
        Args:
            docs (List[Document]): List of documents to summarize
            
        Returns:
            str: Generated summary
        """
        try:
            print("🔄 Starting simple summarization (Stuff method)")
            chain = create_stuff_documents_chain(self.llm, self.stuff_prompt)
            result = chain.invoke({"context": docs})
            print("✅ Simple summarization completed")
            return result
            
        except Exception as e:
            print(f"❌ Simple summarization failed: {e}")
            raise
    
    def _length_function(self, documents: List[Document]) -> int:
        """
        Calculate total token count for a list of documents.
        
        Args:
            documents (List[Document]): List of documents to calculate tokens for
            
        Returns:
            int: Total token count
        """
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)
    
    async def _generate_summary(self, state: SummaryState) -> dict:
        """
        Generate summary for individual document chunk.
        
        Args:
            state (SummaryState): State containing content to summarize
            
        Returns:
            dict: State update with generated summary
        """
        try:
            # Try to use the hub prompt with 'docs' variable first
            prompt = self.map_prompt.invoke({"docs": state["content"]})
            
            response = await self.llm.ainvoke(prompt)
            return {"summaries": [response.content]}
        except Exception as e:
            print(f"❌ Summary generation failed: {e}")
            return {"summaries": [f"Summary generation failed: {str(e)}"]}
    
    def _map_summaries(self, state: OverallState) -> List[Send]:
        """
        Map summary tasks for each content.
        
        Args:
            state (OverallState): Overall state
            
        Returns:
            List[Send]: List of summary tasks for parallel execution
        """
        return [
            Send("generate_summary", {"content": content}) 
            for content in state["contents"]
        ]
    
    def _collect_summaries(self, state: OverallState) -> dict:
        """
        Collect generated summaries and convert them to Document format.
        
        Args:
            state (OverallState): Overall state
            
        Returns:
            dict: State update with collected summaries
        """
        return {
            "collapsed_summaries": [
                Document(page_content=summary) for summary in state["summaries"]
            ]
        }
    
    async def _reduce(self, docs: List[Document]) -> str:
        """
        Combine multiple summaries into one unified summary.
        
        Args:
            docs (List[Document]): List of summary documents to combine
            
        Returns:
            str: Combined summary
        """
        try:
            docs_content = "\n\n".join([doc.page_content for doc in docs])
            prompt = self.reduce_prompt.invoke({"docs": docs_content})
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ Summary reduction failed: {e}")
            return f"Summary reduction failed: {str(e)}"
    
    async def _collapse_summaries(self, state: OverallState) -> dict:
        """
        Collapse summaries when token count exceeds limit.
        
        Args:
            state (OverallState): Overall state
            
        Returns:
            dict: State update with collapsed summaries
        """
        try:
            doc_lists = split_list_of_docs(
                state["collapsed_summaries"], 
                self._length_function, 
                self.config.token_max
            )
            results = []
            for doc_list in doc_lists:
                result = await acollapse_docs(doc_list, self._reduce)
                # Ensure result is a string before creating Document
                if isinstance(result, str):
                    results.append(Document(page_content=result))
                else:
                    # If result is already a Document, extract the content
                    content = result.page_content if hasattr(result, 'page_content') else str(result)
                    results.append(Document(page_content=content))
            
            print(f"📝 Collapsed {len(state['collapsed_summaries'])} summaries into {len(results)} summaries")
            return {"collapsed_summaries": results}
        except Exception as e:
            print(f"❌ Summary collapse failed: {e}")
            return {"collapsed_summaries": state["collapsed_summaries"]}
    
    def _should_collapse(self, state: OverallState) -> str:
        """
        Decide whether to collapse summaries or generate final summary.
        
        Args:
            state (OverallState): Overall state
            
        Returns:
            str: Next step name
        """
        num_tokens = self._length_function(state["collapsed_summaries"])
        if num_tokens > self.config.token_max:
            print(f"⚠️  Token count exceeded ({num_tokens} > {self.config.token_max}), proceeding with collapse")
            return "collapse_summaries"
        else:
            print(f"✅ Token count acceptable ({num_tokens} <= {self.config.token_max}), generating final summary")
            return "generate_final_summary"
    
    async def _generate_final_summary(self, state: OverallState) -> dict:
        """
        Generate final consolidated summary.
        
        Args:
            state (OverallState): Overall state
            
        Returns:
            dict: State update with final summary
        """
        try:
            response = await self._reduce(state["collapsed_summaries"])
            return {"final_summary": response}
        except Exception as e:
            print(f"❌ Final summary generation failed: {e}")
            return {"final_summary": f"Final summary generation failed: {str(e)}"}
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow.
        
        Returns:
            StateGraph: Constructed graph
        """
        print("🔧 Building LangGraph workflow...")
        
        graph = StateGraph(OverallState)
        
        # Add nodes
        graph.add_node("generate_summary", self._generate_summary)
        graph.add_node("collect_summaries", self._collect_summaries)
        graph.add_node("collapse_summaries", self._collapse_summaries)
        graph.add_node("generate_final_summary", self._generate_final_summary)
        
        # Add edges
        graph.add_conditional_edges(START, self._map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self._should_collapse)
        graph.add_conditional_edges("collapse_summaries", self._should_collapse)
        graph.add_edge("generate_final_summary", END)
        
        print("✅ LangGraph workflow construction completed")
        return graph
    
    async def summarize_with_mapreduce(self, docs: List[Document]) -> str:
        """
        Summarize documents using Map-Reduce approach.
        
        Args:
            docs (List[Document]): List of documents to summarize
            
        Returns:
            str: Generated final summary
            
        Raises:
            Exception: If summarization process fails
        """
        try:
            print("🚀 === Starting Map-Reduce Summarization Pipeline ===")
            print(f"📊 Number of document chunks to process: {len(docs)}")
            
            # Build and compile graph
            if not self.app:
                graph = self._build_graph()
                self.app = graph.compile()
            
            # Stream execution
            final_summary = ""
            async for step in self.app.astream(
                {"contents": [doc.page_content for doc in docs]},
                {"recursion_limit": self.config.recursion_limit},
            ):
                step_names = list(step.keys())
                print(f"⚡ Execution step: {step_names}")
                
                # Extract final summary
                if "generate_final_summary" in step:
                    if "final_summary" in step["generate_final_summary"]:
                        final_summary = step["generate_final_summary"]["final_summary"]
            
            print("🎉 === Map-Reduce Summarization Pipeline Completed ===")
            return final_summary
            
        except Exception as e:
            print(f"❌ Map-Reduce summarization failed: {e}")
            raise


async def main():
    """
    Main execution function.
    
    Runs the document summarization pipeline to perform both simple and Map-Reduce summarization.
    """
    try:
        # Create configuration
        config = SummarizationConfig(
            model_name="gpt-4o-mini",
            temperature=0.0,
            chunk_size=1000,
            token_max=1000
        )
        
        # Initialize summarizer
        summarizer = DocumentSummarizer(config)
        
        # Load documents
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        docs = summarizer.load_documents(url)
        
        print("\n" + "="*60)
        print("📄 Document Summarization Pipeline")
        print("="*60)
        
        # 1. Simple summarization (Stuff method)
        print("\n🔄 1. Simple Summarization (Stuff Method)")
        print("-" * 40)
        try:
            simple_summary = summarizer.simple_summarize(docs)
            print("✅ Simple Summary:")
            print(simple_summary[:500] + "..." if len(simple_summary) > 500 else simple_summary)
        except Exception as e:
            print(f"❌ Simple summarization failed: {e}")
        
        # 2. Map-Reduce summarization
        print(f"\n🔄 2. Map-Reduce Summarization ({len(docs)} chunks)")
        print("-" * 40)
        mapreduce_summary = await summarizer.summarize_with_mapreduce(docs)
        
        print("✅ Map-Reduce Summary:")
        print(mapreduce_summary)
        
        print("\n" + "="*60)
        print("✅ Summarization Pipeline Completed Successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())