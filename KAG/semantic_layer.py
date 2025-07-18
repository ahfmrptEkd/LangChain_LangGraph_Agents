from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

def initialize_graph():
    """
    Neo4j graph database initialization and loading movie data
    
    Returns:
        Neo4jGraph: Initialized Neo4j graph object
    """
    graph = Neo4jGraph(refresh_schema=False)
    
    movies_query = """
    LOAD CSV WITH HEADERS FROM 
    'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
    AS row
    MERGE (m:Movie {id:row.movieId})
    SET m.released = date(row.released),
        m.title = row.title,
        m.imdbRating = toFloat(row.imdbRating)
    FOREACH (director in split(row.director, '|') | 
        MERGE (p:Person {name:trim(director)})
        MERGE (p)-[:DIRECTED]->(m))
    FOREACH (actor in split(row.actors, '|') | 
        MERGE (p:Person {name:trim(actor)})
        MERGE (p)-[:ACTED_IN]->(m))
    FOREACH (genre in split(row.genres, '|') | 
        MERGE (g:Genre {name:trim(genre)})
        MERGE (m)-[:IN_GENRE]->(g))
    """
    
    graph.query(movies_query)
    return graph

def get_information(entity: str, graph: Neo4jGraph) -> str:
    """
    Search for information about movies or people in the Neo4j graph
    
    Args:
        entity (str): The name of the movie or person to search for
        graph (Neo4jGraph): Neo4j graph object
        
    Returns:
        str: The found information or "No information was found"
    """
    description_query = """
    MATCH (m:Movie|Person)
    WHERE m.title CONTAINS $candidate OR m.name CONTAINS $candidate
    MATCH (m)-[r:ACTED_IN|IN_GENRE]-(t)
    WITH m, type(r) as type, collect(coalesce(t.name, t.title)) as names
    WITH m, type+": "+reduce(s="", n IN names | s + n + ", ") as types
    WITH m, collect(types) as contexts
    WITH m, "type:" + labels(m)[0] + "\ntitle: "+ coalesce(m.title, m.name) 
           + "\nyear: "+coalesce(m.released,"") +"\n" +
           reduce(s="", c in contexts | s + substring(c, 0, size(c)-2) +"\n") as context
    RETURN context LIMIT 1
    """
    
    try:
        data = graph.query(description_query, params={"candidate": entity})
        return data[0]["context"]
    except IndexError:
        return "No information was found"

class InformationInput(BaseModel):
    """Input schema for the information search tool"""
    entity: str = Field(description="movie or a person mentioned in the question")

class InformationTool(BaseTool):
    """
    Tool class for searching for information about movies and people
    
    This tool searches for information about movies or people in the Neo4j graph database.
    """
    name: str = "information"
    description: str = ("useful for when you need to answer questions about various actors or movies")
    args_schema: Type[BaseModel] = InformationInput
    graph: Neo4jGraph

    def __init__(self, graph: Neo4jGraph):
        super().__init__(graph=graph)

    def _run(self, entity: str) -> str:
        """Use the tool."""
        return get_information(entity, self.graph)

    def _arun(self, entity: str) -> str:
        """Use the tool asynchronously."""
        return self._run(entity)

def create_workflow(graph: Neo4jGraph):
    """
    Create a workflow for searching for movie information
    
    Args:
        graph (Neo4jGraph): Neo4j graph object
        
    Returns:
        StateGraph: Compiled workflow graph
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    tools = [InformationTool(graph)]
    llm_with_tools = llm.bind_tools(tools)
    
    system_message = SystemMessage(
        content="You are a helpful assistant tasked with finding and explaining relevant information about movies."
    )
    
    def assistant(state: MessagesState):
        """Assistant node function"""
        return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}
    
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges(
        "assistant",

        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    workflow.add_edge("tools", "assistant")
    
    return workflow.compile()

def main():
    """
    메인 함수 - 영화 정보 검색 시스템을 실행
    
    이 함수는 다음과 같은 작업을 수행합니다:
    1. Neo4j 그래프 데이터베이스 초기화 및 데이터 로드
    2. 워크플로우 생성
    3. 사용자 질의 처리 및 결과 출력
    """
    graph = initialize_graph()
    
    react_graph = create_workflow(graph)
    
    input_messages = [HumanMessage(content="Who played in the Casino?")]
    messages = react_graph.invoke({"messages": input_messages})
    
    for m in messages["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    main()