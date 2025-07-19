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

# TODO: recommend tool 만들기. (1. 같은 장르, 고평점 영화 추천, 2. 특정 배우의 다른 영화 (장르) 추천, 3. 사용자가 좋아한 영화 기반 추천)
def recommend_by_genre(movie_title: str, graph: Neo4jGraph, limit: int = 5) -> str:
    """
    Recommend high-rated movies in the same genre
    
    Args:
        movie_title (str): The title of the reference movie
        graph (Neo4jGraph): Neo4j graph object
        limit (int): Number of recommendations to return
        
    Returns:
        str: Recommended movies or "No recommendations found"
    """
    genre_recommendation_query = """
    MATCH (m:Movie {title: $movie_title})-[:IN_GENRE]->(g:Genre)
    MATCH (rec:Movie)-[:IN_GENRE]->(g)
    WHERE rec.title <> m.title AND rec.imdbRating IS NOT NULL
    WITH rec, g.name as genre, rec.imdbRating as rating
    ORDER BY rating DESC
    WITH rec, collect(DISTINCT genre) as genres, rating
    RETURN "Title: " + rec.title + 
           "\nYear: " + coalesce(toString(rec.released), "N/A") +
           "\nRating: " + toString(rating) +
           "\nGenres: " + reduce(s="", g in genres | s + g + ", ") +
           "\n---" as recommendation
    LIMIT $limit
    """
    
    try:
        data = graph.query(genre_recommendation_query, 
                          params={"movie_title": movie_title, "limit": limit})
        if data:
            recommendations = "\n".join([item["recommendation"] for item in data])
            return f"Movies in similar genres:\n\n{recommendations}"
        else:
            return "No recommendations found"
    except Exception as e:
        return f"Error finding recommendations: {str(e)}"
    
def recommend_by_actor(actor_name: str, graph: Neo4jGraph, limit: int = 5) -> str:
    """
    Recommend high-rated movies with the same actors
    
    Args:
        actor_name (str): The name of the reference actor
        graph (Neo4jGraph): Neo4j graph object
        limit (int): Number of recommendations to return
        
    Returns:
        str: Recommended movies or "No recommendations found"
    """
    actor_recommendation_query = """
    MATCH (m:Movie {title: $actor_name})<-[:ACTED_IN]-(actor:Person)
    MATCH (actor)-[:ACTED_IN]->(rec:Movie)
    WHERE rec.title <> m.title AND rec.imdbRating IS NOT NULL
    WITH rec, collect(DISTINCT actor.name) as shared_actors, rec.imdbRating as rating
    ORDER BY rating DESC, size(shared_actors) DESC
    RETURN "Title: " + rec.title + 
           "\nYear: " + coalesce(toString(rec.released), "N/A") +
           "\nRating: " + toString(rating) +
           "\nShared Actors: " + reduce(s="", a in shared_actors | s + a + ", ") +
           "\n---" as recommendation
    LIMIT $limit
    """
    
    try:
        data = graph.query(actor_recommendation_query, 
                          params={"actor_name": actor_name, "limit": limit})
        if data:
            recommendations = "\n".join([item["recommendation"] for item in data])
            return f"Movies with same actors:\n\n{recommendations}"
        else:
            return "No recommendations found"
    except Exception as e:
        return f"Error finding recommendations: {str(e)}"
    
def recommend_personalized(liked_movies: list, graph: Neo4jGraph, limit: int = 5) -> str:
    """
    Personalized movie recommendations based on user's liked movies
    
    Args:
        liked_movies (list): List of movie titles the user liked
        graph (Neo4jGraph): Neo4j graph object
        limit (int): Number of recommendations to return
        
    Returns:
        str: Personalized recommendations or "No recommendations found"
    """
    personalized_query = """
    // Get genres and actors from liked movies
    MATCH (liked:Movie)
    WHERE liked.title IN $liked_movies
    
    // Collect preferred genres
    OPTIONAL MATCH (liked)-[:IN_GENRE]->(g:Genre)
    WITH collect(DISTINCT g.name) as preferred_genres, liked
    
    // Collect preferred actors
    OPTIONAL MATCH (liked)<-[:ACTED_IN]-(a:Person)
    WITH preferred_genres, collect(DISTINCT a.name) as preferred_actors, collect(DISTINCT liked.title) as liked_titles
    
    // Find recommendations based on preferences
    MATCH (rec:Movie)
    WHERE NOT rec.title IN liked_titles AND rec.imdbRating IS NOT NULL
    
    // Calculate genre score
    OPTIONAL MATCH (rec)-[:IN_GENRE]->(rg:Genre)
    WHERE rg.name IN preferred_genres
    WITH rec, preferred_actors, liked_titles, count(DISTINCT rg) as genre_score
    
    // Calculate actor score
    OPTIONAL MATCH (rec)<-[:ACTED_IN]-(ra:Person)
    WHERE ra.name IN preferred_actors
    WITH rec, genre_score, count(DISTINCT ra) as actor_score, rec.imdbRating as rating
    
    // Calculate total recommendation score
    WITH rec, 
         (genre_score * 2 + actor_score * 3 + rating) as rec_score,
         genre_score, actor_score, rating
    WHERE genre_score > 0 OR actor_score > 0
    
    ORDER BY rec_score DESC, rating DESC
    
    RETURN "Title: " + rec.title + 
           "\nYear: " + coalesce(toString(rec.released), "N/A") +
           "\nRating: " + toString(rating) +
           "\nRecommendation Score: " + toString(round(rec_score * 100) / 100) +
           "\n---" as recommendation
    LIMIT $limit
    """
    
    try:
        data = graph.query(personalized_query, 
                          params={"liked_movies": liked_movies, "limit": limit})
        if data:
            recommendations = "\n".join([item["recommendation"] for item in data])
            return f"Personalized recommendations based on your preferences:\n\n{recommendations}"
        else:
            return "No personalized recommendations found"
    except Exception as e:
        return f"Error finding personalized recommendations: {str(e)}"

# Input schemas for different recommendation tools
class RecommendGenreInput(BaseModel):
    """Input schema for genre-based recommendations"""
    movie_title: str = Field(description="movie title to find similar genre movies for")

class RecommendActorInput(BaseModel):
    """Input schema for actor-based recommendations"""
    actor_name: str = Field(description="actor name to find movies with same actors")

class RecommendPersonalizedInput(BaseModel):
    """Input schema for personalized recommendations"""
    liked_movies: str = Field(description="comma-separated list of movie titles the user liked")

class RecommendGenreTool(BaseTool):
    """Tool for recommending movies of the same genre"""
    name: str = "recommend_genre"
    description: str = "useful for when you need to recommend movies of the same genre as a given movie"
    args_schema: Type[BaseModel] = RecommendGenreInput
    graph: Neo4jGraph

    def __init__(self, graph: Neo4jGraph):
        super().__init__(graph=graph)

    def _run(self, movie_title: str) -> str:
        """Use the tool."""
        return recommend_by_genre(movie_title, self.graph, 5)
    
    def _arun(self, movie_title: str) -> str:
        """Use the tool asynchronously."""
        return self._run(movie_title)

class RecommendActorTool(BaseTool):
    """Tool for recommending movies with the same actors"""
    name: str = "recommend_actor"
    description: str = "useful for when you need to recommend movies with the same actors as a given movie"
    args_schema: Type[BaseModel] = RecommendActorInput
    graph: Neo4jGraph

    def __init__(self, graph: Neo4jGraph):
        super().__init__(graph=graph)

    def _run(self, actor_name: str) -> str:
        """Use the tool."""
        return recommend_by_actor(actor_name, self.graph, 5)
    
    def _arun(self, actor_name: str) -> str:
        """Use the tool asynchronously."""
        return self._run(actor_name)
    
class RecommendPersonalizedTool(BaseTool):
    """Tool for personalized movie recommendations"""
    name: str = "recommend_personalized"
    description: str = "useful for when you need to recommend movies based on the user's favorite movies"
    args_schema: Type[BaseModel] = RecommendPersonalizedInput
    graph: Neo4jGraph
    
    def __init__(self, graph: Neo4jGraph):
        super().__init__(graph=graph)
        
    def _run(self, liked_movies: str) -> str:
        """Use the tool."""
        # Convert comma-separated string to list
        liked_movies_list = [movie.strip() for movie in liked_movies.split(",")]
        return recommend_personalized(liked_movies_list, self.graph, 5)
    
    def _arun(self, liked_movies: str) -> str:
        """Use the tool asynchronously."""
        return self._run(liked_movies)

def create_workflow(graph: Neo4jGraph):
    """
    Create a workflow for searching for movie information
    
    Args:
        graph (Neo4jGraph): Neo4j graph object
        
    Returns:
        StateGraph: Compiled workflow graph
    """
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    tools = [InformationTool(graph), RecommendGenreTool(graph), RecommendActorTool(graph), RecommendPersonalizedTool(graph)]
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
    
    input_questions = [
        # "Who played in the Casino?",
        # "Recommend same genre movies as Toy Story",
        "Recommend movies with Tom Hanks",
        "What movies would you recommend to someone who likes Father of the Bride Part II, Grumpier Old Men, and Waiting to Exhale?"
    ]

    for question in input_questions:
        input_messages = [HumanMessage(content=question)]
        messages = react_graph.invoke({"messages": input_messages})

        for m in messages["messages"]:
            m.pretty_print()
        print("-" * 100)

if __name__ == "__main__":
    main()