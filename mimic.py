import json
import networkx as nx
import random
import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class Mimic:
    """
    Mimic ingests the knowledge graph and learns to generate new content 
    based on the patterns discovered - like a parasite mimicking its host
    to infiltrate and reproduce.
    """
    
    def __init__(self, input_graph=None, output_dir=None):
        """
        Initialize the Mimic generator.
        
        Args:
            input_graph (str): Path to the JSON knowledge graph file
            output_dir (str): Directory to save generated content
        """
        import os
        
        # Set default paths within the carnis_data directory structure
        if input_graph is None:
            input_graph = os.path.join('carnis_data', 'meatsnake', 'knowledge_graph.json')
        
        if output_dir is None:
            # Ensure the mimicked_content directory exists
            output_dir = os.path.join('carnis_data', 'mimic')
            os.makedirs(output_dir, exist_ok=True)

        self.input_graph = input_graph
        self.output_dir = output_dir
        self.knowledge_graph = None
        self.entity_embeddings = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set up the language model for generation
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
            print("GPT-2 model loaded successfully.")
        except Exception as e:
            print(f"Error loading GPT-2 model: {e}")
            print("Will use rule-based generation as fallback.")
            self.tokenizer = None
            self.model = None
    
    def load_graph(self):
        """Load the knowledge graph from JSON file"""
        try:
            with open(self.input_graph, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            self.knowledge_graph = nx.node_link_graph(graph_data)
            print(f"Knowledge graph loaded with {self.knowledge_graph.number_of_nodes()} nodes and "
                  f"{self.knowledge_graph.number_of_edges()} edges.")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.input_graph} not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: File {self.input_graph} contains invalid JSON.")
            return False
    
    def analyze_graph_patterns(self):
        """Analyze patterns in the knowledge graph to inform generation"""
        if not self.knowledge_graph:
            if not self.load_graph():
                return {}
        
        patterns = {
            'common_relations': Counter(),
            'entity_frequencies': Counter(),
            'concept_clusters': {},
            'topic_distribution': Counter()
        }
        
        # Extract common relations
        for _, _, data in self.knowledge_graph.edges(data=True):
            relation = data.get('relation', 'connected')
            patterns['common_relations'][relation] += 1
        
        # Entity frequencies
        for node, data in self.knowledge_graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            label = data.get('label', node)
            if node_type == 'entity':
                patterns['entity_frequencies'][label] += 1
            category = data.get('category', 'unknown')
            patterns['topic_distribution'][category] += 1
        
        # Find clusters of concepts based on connectivity
        # This is a simplified community detection
        try:
            if len(self.knowledge_graph) > 0:
                clusters = nx.community.greedy_modularity_communities(
                    self.knowledge_graph.to_undirected())
                for i, cluster in enumerate(clusters):
                    patterns['concept_clusters'][f'cluster_{i}'] = list(cluster)
        except Exception as e:
            print(f"Warning: Could not detect communities: {e}")
        
        return patterns
    
    def create_walk_paths(self, num_paths=20, max_length=10):
        """Create random walk paths through the knowledge graph to capture relationships"""
        if not self.knowledge_graph:
            if not self.load_graph():
                return []
        
        paths = []
        nodes = list(self.knowledge_graph.nodes())
        
        for _ in range(num_paths):
            if not nodes:
                break
                
            # Start from a random node
            current = random.choice(nodes)
            path = [current]
            
            # Walk the graph
            for _ in range(max_length - 1):
                neighbors = list(self.knowledge_graph.neighbors(current))
                if not neighbors:
                    break
                    
                # Select a neighbor
                current = random.choice(neighbors)
                path.append(current)
            
            # Convert node IDs to labels
            labeled_path = []
            for node_id in path:
                node_data = self.knowledge_graph.nodes[node_id]
                label = node_data.get('label', node_id)
                labeled_path.append(label)
            
            paths.append(labeled_path)
        
        return paths
    
    def generate_text_prompt(self, seed_concepts=None, max_length=200):
        """Generate text prompts based on the knowledge graph"""
        if not self.knowledge_graph:
            if not self.load_graph():
                return "Failed to load knowledge graph for text generation."
        
        # Find important concepts to include in the prompt
        if not seed_concepts:
            # Get nodes with highest degree centrality
            centrality = nx.degree_centrality(self.knowledge_graph)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            seed_concepts = []
            for node_id, _ in top_nodes:
                node_data = self.knowledge_graph.nodes[node_id]
                if 'label' in node_data:
                    seed_concepts.append(node_data['label'])
        
        # Create a prompt based on these concepts
        prompt_parts = [
            "Based on the following interconnected concepts:",
            ", ".join(seed_concepts),
            "Generate a coherent narrative that explores their relationships.",
        ]
        
        prompt = " ".join(prompt_parts)
        
        return prompt
    
    def generate_text(self, prompt, max_length=300):
        """Generate text using GPT-2 based on a prompt"""
        if self.model is None or self.tokenizer is None:
            return self._fallback_generate(prompt, max_length)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate text
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.85
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text
        except Exception as e:
            print(f"Error generating text with GPT-2: {e}")
            return self._fallback_generate(prompt, max_length)
    
    def _fallback_generate(self, prompt, max_length=300):
        """Fallback text generation when GPT-2 is not available"""
        # Analyze patterns for rule-based generation
        patterns = self.analyze_graph_patterns()
        common_relations = patterns.get('common_relations', Counter())
        entity_frequencies = patterns.get('entity_frequencies', Counter())
        
        # Get most common entities and relations
        top_entities = [e for e, _ in entity_frequencies.most_common(20)]
        top_relations = [r for r, _ in common_relations.most_common(10)]
        
        # Create placeholder sentences
        sentences = []
        for _ in range(5):
            if top_entities and top_relations:
                entity1 = random.choice(top_entities)
                entity2 = random.choice(top_entities)
                relation = random.choice(top_relations)
                
                sentence = f"{entity1} {relation} {entity2}."
                sentences.append(sentence)
        
        # Add the prompt at the beginning
        result = prompt + "\n\n" + " ".join(sentences)
        
        return result
    
    def visualize_mimicry(self, num_samples=5):
        """Visualize the mimicry pattern emerging from the graph"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend that's thread-safe
        import matplotlib.pyplot as plt
        
        if not self.knowledge_graph:
            if not self.load_graph():
                return
        
        # Create output visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Create a subgraph for visualization
        central_nodes = sorted(
            nx.degree_centrality(self.knowledge_graph).items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        central_node_ids = [node_id for node_id, _ in central_nodes]
        subgraph = self.knowledge_graph.subgraph(central_node_ids)
        
        # Create different views of the subgraph to mimic evolution
        plt.figure(figsize=(12, 10))
        
        # Spring layout for consistent positioning
        pos = nx.spring_layout(subgraph, seed=42)
        
        for i in range(num_samples):
            plt.clf()
            
            # Get node types for coloring
            node_types = [subgraph.nodes[node].get('type', 'unknown') for node in subgraph]
            node_colors = ['#3388AA' if t == 'document' else '#CC6677' if t == 'entity' else '#44BB99' 
                          for t in node_types]
            
            # Random edge subset to show evolution
            edge_subset = random.sample(list(subgraph.edges()), 
                                        k=int(subgraph.number_of_edges() * (i+1)/num_samples))
            edge_subgraph = nx.DiGraph()
            edge_subgraph.add_nodes_from(subgraph.nodes(data=True))
            edge_subgraph.add_edges_from([(u, v, subgraph.edges[u, v]) for u, v in edge_subset])
            
            # Draw the graph
            nx.draw_networkx(
                edge_subgraph, pos,
                with_labels=True, 
                node_color=node_colors,
                node_size=300,
                font_size=8,
                width=0.8,
                alpha=0.8,
                arrows=True
            )
            
            plt.title(f"Mimicry Evolution - Stage {i+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"mimicry_stage_{i+1}.png"))
        
        print(f"Mimicry visualization saved to {viz_dir}")
    
    def generate_mimicked_content(self, num_samples=3):
        """Generate multiple content samples by mimicking the knowledge graph patterns"""
        if not self.knowledge_graph:
            if not self.load_graph():
                return []
        
        generated_samples = []
        
        # Analyze the graph to get patterns
        patterns = self.analyze_graph_patterns()
        
        # Generate samples
        for i in range(num_samples):
            print(f"Generating mimicked content sample {i+1}/{num_samples}...")
            
            # Get walk paths to capture narrative flow
            walk_paths = self.create_walk_paths(num_paths=5, max_length=5)
            
            # Extract key concepts from random walk paths
            concepts = set()
            for path in walk_paths:
                concepts.update(path)
            
            # Convert to list and limit
            seed_concepts = list(concepts)[:7]
            
            # Generate text prompt
            prompt = self.generate_text_prompt(seed_concepts)
            
            # Generate text content
            content = self.generate_text(prompt)
            
            # Create sample object
            sample = {
                'id': i + 1,
                'seed_concepts': seed_concepts,
                'prompt': prompt,
                'content': content,
                'paths': walk_paths
            }
            
            generated_samples.append(sample)
            
            # Save individual sample
            with open(os.path.join(self.output_dir, f"mimicked_sample_{i+1}.json"), 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2)
        
        # Save all samples
        with open(os.path.join(self.output_dir, "all_mimicked_samples.json"), 'w', encoding='utf-8') as f:
            json.dump(generated_samples, f, indent=2)
        
        print(f"Generated {len(generated_samples)} mimicked content samples.")
        return generated_samples

if __name__ == "__main__":
    # Example usage with default paths that use the carnis_data directory structure
    mimic = Mimic()
    mimic.generate_mimicked_content(num_samples=3)
    mimic.visualize_mimicry()