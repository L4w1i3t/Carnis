import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import matplotlib.pyplot as plt
from pyvis.network import Network
import os
import subprocess
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Meatsnake:
    """
    Meatsnake consumes trimmed data and transforms it into a knowledge graph and vector store,
    creating connections between concepts - like a snake digesting meat
    and building new structures from it.
    """

    def __init__(self, input_file=None, output_file=None, vector_store_dir=None):
        """
        Initialize the Meatsnake processor.
        
        Args:
            input_file (str): Path to the JSON file containing trimmed data
            output_file (str): Path to save the knowledge graph data
            vector_store_dir (str): Directory to save the vector store
        """
        import os

        # Set default paths within the carnis_data directory structure
        if input_file is None:
            input_file = os.path.join('carnis_data', 'trimmings', 'trimmed_data.json')

        if output_file is None:
            # Ensure the knowledge directory exists
            os.makedirs(os.path.join('carnis_data', 'meatsnake'), exist_ok=True)
            output_file = os.path.join('carnis_data', 'meatsnake', 'knowledge_graph.json')

        if vector_store_dir is None:
            vector_store_dir = os.path.join('carnis_data', 'meatsnake', 'vector_store')
            os.makedirs(vector_store_dir, exist_ok=True)

        self.input_file = input_file
        self.output_file = output_file
        self.vector_store_dir = vector_store_dir
        self.trimmed_data = []
        self.knowledge_graph = nx.DiGraph()

        # Initialize RAG components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Load spaCy NLP model for entity recognition and dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
            except subprocess.CalledProcessError:
                print("Failed to download model automatically. Please run:")
                print("python -m spacy download en_core_web_sm")
                sys.exit(1)

        # Vector store placeholder
        self.vector_store = None

    def load_data(self):
        """Load data from the input JSON file"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.trimmed_data = json.load(f)
            print(f"Loaded {len(self.trimmed_data)} documents for processing.")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.input_file} not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: File {self.input_file} contains invalid JSON.")
            return False

    def extract_entities(self, text):
        """
        Extract named entities and key concepts from text.
        
        Args:
            text (str): Text to process
            
        Returns:
            list: Extracted entities and concepts
        """
        doc = self.nlp(text)

        # Extract named entities
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "type": "entity"
                })

        # Extract key noun phrases as concepts
        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Filter out very short noun phrases or those that are just pronouns
            if len(chunk.text) > 3 and not chunk.root.pos_ == "PRON":
                noun_phrases.append({
                    "text": chunk.text,
                    "label": "CONCEPT",
                    "type": "concept"
                })

        # Combine and deduplicate
        all_items = entities + noun_phrases
        unique_items = []
        seen = set()
        for item in all_items:
            if item["text"].lower() not in seen:
                seen.add(item["text"].lower())
                unique_items.append(item)

        return unique_items

    def extract_relations(self, text):
        """
        Extract semantic relationships between entities and concepts.
        
        Args:
            text (str): Text to process
            
        Returns:
            list: Extracted relationships
        """
        doc = self.nlp(text)
        relations = []

        # Extract subject-verb-object relations
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                subj = token.text
                verb = token.head.text

                # Find objects of the verb
                for child in token.head.children:
                    if child.dep_ in ("dobj", "pobj", "attr"):
                        obj = child.text
                        relations.append({
                            "source": subj,
                            "relation": verb,
                            "target": obj
                        })

        return relations

    def build_graph(self):
        """Build knowledge graph from trimmed data"""
        if not self.load_data():
            return None

        # Clear existing graph
        self.knowledge_graph = nx.DiGraph()

        # Process each document
        all_entities = {}
        all_relations = []

        for idx, item in enumerate(self.trimmed_data):
            print(f"Processing document {idx+1}/{len(self.trimmed_data)}: {item['title']}")

            # Extract entities from summary
            entities = self.extract_entities(item["summary"])

            # Add document title as a source node
            doc_node_id = f"doc:{idx}"
            self.knowledge_graph.add_node(doc_node_id, 
                                         label=item["title"],
                                         type="document",
                                         url=item["url"])

            # Add entities to graph and link to document
            for entity in entities:
                entity_id = f"{entity['label']}:{entity['text'].lower()}"

                # Add entity node if it doesn't exist
                if entity_id not in all_entities:
                    all_entities[entity_id] = entity
                    self.knowledge_graph.add_node(entity_id,
                                                label=entity['text'],
                                                type=entity['type'],
                                                category=entity['label'])

                # Connect document to entity
                self.knowledge_graph.add_edge(doc_node_id, entity_id, relation="contains")

            # Extract relations from summary
            relations = self.extract_relations(item["summary"])

            # Add relations to collection (will process later)
            for relation in relations:
                relation["doc_id"] = idx
                all_relations.append(relation)

        # Process relations and add to graph
        for relation in all_relations:
            # Look for source and target in existing nodes
            source_candidates = [node for node in self.knowledge_graph.nodes() 
                                if relation["source"].lower() in node.lower()]
            target_candidates = [node for node in self.knowledge_graph.nodes() 
                                if relation["target"].lower() in node.lower()]

            if source_candidates and target_candidates:
                # Use the first match for simplicity
                source = source_candidates[0]
                target = target_candidates[0]

                # Add relationship
                self.knowledge_graph.add_edge(source, target, 
                                            relation=relation["relation"],
                                            doc_id=relation["doc_id"])

        # Find connections between similar entities using keywords
        self.find_keyword_connections()

        # Save graph data
        self.save_graph()

        # Build vector store for RAG
        self.build_vector_store()

        return self.knowledge_graph

    def find_keyword_connections(self):
        """Connect nodes that share significant keywords"""
        # Create a mapping of documents to their keywords
        doc_keywords = {}
        for idx, item in enumerate(self.trimmed_data):
            doc_id = f"doc:{idx}"
            doc_keywords[doc_id] = set(item["keywords"])

        # Connect documents that share significant keywords
        doc_ids = list(doc_keywords.keys())
        for i in range(len(doc_ids)):
            for j in range(i+1, len(doc_ids)):
                doc1 = doc_ids[i]
                doc2 = doc_ids[j]

                # Calculate Jaccard similarity of keywords
                common_keywords = doc_keywords[doc1].intersection(doc_keywords[doc2])
                all_keywords = doc_keywords[doc1].union(doc_keywords[doc2])

                if all_keywords:
                    similarity = len(common_keywords) / len(all_keywords)

                    # Connect if similarity is significant
                    if similarity > 0.2 and common_keywords:
                        self.knowledge_graph.add_edge(
                            doc1, doc2, 
                            relation="related", 
                            similarity=round(similarity, 2),
                            common_keywords=list(common_keywords)
                        )

    def save_graph(self):
        """Save the knowledge graph to file and create visualization"""
        import os

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Save as JSON for later use
        graph_data = nx.node_link_data(self.knowledge_graph)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)

        # Create HTML visualization in the same directory
        html_file = self.output_file.replace('.json', '.html')
        self.visualize_graph(html_file)

        print(f"Knowledge graph built with {self.knowledge_graph.number_of_nodes()} nodes and "
            f"{self.knowledge_graph.number_of_edges()} edges.")
        print(f"Graph data saved to {self.output_file}")
        print(f"Graph visualization saved to {html_file}")

    def visualize_graph(self, output_file):
        """Create an interactive visualization of the graph"""
        # Create a PyVis network
        net = Network(height="750px", width="100%", notebook=False, directed=True)

        # Add nodes with different colors based on type
        color_map = {
            "document": "#3388AA",
            "entity": "#CC6677",
            "concept": "#44BB99"
        }

        for node_id in self.knowledge_graph.nodes():
            node_data = self.knowledge_graph.nodes[node_id]
            node_type = node_data.get("type", "concept")

            net.add_node(
                node_id, 
                label=node_data.get("label", node_id),
                title=f"{node_data.get('label', node_id)} ({node_data.get('category', 'N/A')})",
                color=color_map.get(node_type, "#BBBBBB")
            )

        # Add edges
        for source, target, data in self.knowledge_graph.edges(data=True):
            net.add_edge(
                source, target, 
                title=data.get("relation", "connected"),
                label=data.get("relation", "")
            )

        # Set physics options for better visualization
        net.set_options("""
        const options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 25
            }
          }
        }
        """)

        # Save visualization
        net.save_graph(output_file)

    def build_vector_store(self):
        """
        Build a vector store from the trimmed data for RAG capabilities.
        This complements the knowledge graph by enabling semantic search.
        """
        if not self.trimmed_data:
            if not self.load_data():
                return None

        print("Building vector store for RAG...")
        documents = []

        # Process each document
        for idx, item in enumerate(self.trimmed_data):
            # Extract text and metadata
            text = item.get("content", "")
            if not text and "summary" in item:
                text = item["summary"]

            metadata = {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "source_id": f"doc:{idx}",
                "keywords": item.get("keywords", [])
            }

            # Split text into chunks
            if text:
                text_chunks = self.text_splitter.split_text(text)

                # Add each chunk with its metadata
                for i, chunk in enumerate(text_chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    documents.append({"content": chunk, "metadata": chunk_metadata})

        # Create or update vector store
        if documents:
            if os.path.exists(os.path.join(self.vector_store_dir, 'index.faiss')):
                # Load existing vector store and add new documents
                print("Updating existing vector store...")
                self.vector_store = FAISS.load_local(self.vector_store_dir, self.embeddings)
                self.vector_store.add_texts(
                    [doc["content"] for doc in documents],
                    [doc["metadata"] for doc in documents]
                )
            else:
                # Create new vector store
                print("Creating new vector store...")
                self.vector_store = FAISS.from_texts(
                    [doc["content"] for doc in documents],
                    self.embeddings,
                    metadatas=[doc["metadata"] for doc in documents]
                )

            # Save vector store
            self.vector_store.save_local(self.vector_store_dir)
            print(f"Vector store created with {len(documents)} chunks and saved to {self.vector_store_dir}")
            return self.vector_store
        else:
            print("No documents to process for vector store")
            return None

    def query_vector_store(self, query, k=5):
        """
        Query the vector store to retrieve relevant documents.
        
        Args:
            query (str): The query text
            k (int): Number of documents to retrieve
            
        Returns:
            list: Retrieved documents
        """
        # Load vector store if not already loaded
        if self.vector_store is None:
            if os.path.exists(os.path.join(self.vector_store_dir, 'index.faiss')):
                self.vector_store = FAISS.load_local(self.vector_store_dir, self.embeddings)
            else:
                print("Vector store not found. Run build_vector_store first.")
                return []

        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k)

        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return formatted_results

    def hybrid_query(self, query, k=5):
        """
        Perform a hybrid query using both the knowledge graph and vector store.
        This combines symbolic and semantic search capabilities.
        
        Args:
            query (str): The query text
            k (int): Number of results to retrieve
            
        Returns:
            dict: Combined results from graph and vector store
        """
        # Get results from vector store
        vector_results = self.query_vector_store(query, k=k)

        # Extract entities from query
        query_entities = self.extract_entities(query)

        # Find relevant nodes in knowledge graph
        graph_results = []
        for entity in query_entities:
            entity_text = entity["text"].lower()

            # Find matching nodes
            for node_id, node_data in self.knowledge_graph.nodes(data=True):
                node_label = node_data.get("label", "").lower()

                if entity_text in node_label:
                    # Get connected nodes (1-hop neighborhood)
                    neighbors = list(self.knowledge_graph.neighbors(node_id))

                    # Add node and its neighbors
                    graph_results.append({
                        "node_id": node_id,
                        "label": node_data.get("label", ""),
                        "type": node_data.get("type", ""),
                        "neighbors": [
                            {
                                "node_id": neighbor,
                                "label": self.knowledge_graph.nodes[neighbor].get("label", ""),
                                "relation": self.knowledge_graph.edges[node_id, neighbor].get("relation", "connected")
                            }
                            for neighbor in neighbors
                        ]
                    })

        # Combine results
        return {
            "vector_results": vector_results,
            "graph_results": graph_results
        }

if __name__ == "__main__":
    # Example usage with default paths that use the carnis_data directory structure
    snake = Meatsnake()
    graph = snake.build_graph()