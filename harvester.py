import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import networkx as nx

class Harvester:
    """
    Harvester extracts valuable patterns and insights from mimicked content,
    drawing nutrients and energy from it.
    """
    
    def __init__(self, input_dir='mimicked_content', output_dir='harvested_insights'):
        """
        Initialize the Harvester.
        
        Args:
            input_dir (str): Directory containing mimicked content
            output_dir (str): Directory to save harvested insights
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mimicked_samples = []
        self.harvested_insights = {}
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create visualization subdirectory
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
    
    def load_mimicked_content(self):
        """Load mimicked content samples"""
        try:
            all_samples_file = os.path.join(self.input_dir, "all_mimicked_samples.json")
            if os.path.exists(all_samples_file):
                with open(all_samples_file, 'r', encoding='utf-8') as f:
                    self.mimicked_samples = json.load(f)
                print(f"Loaded {len(self.mimicked_samples)} mimicked content samples.")
                return True
            else:
                # Try loading individual sample files
                sample_files = [f for f in os.listdir(self.input_dir) if f.startswith('mimicked_sample_') and f.endswith('.json')]
                for file in sample_files:
                    with open(os.path.join(self.input_dir, file), 'r', encoding='utf-8') as f:
                        self.mimicked_samples.append(json.load(f))
                
                if self.mimicked_samples:
                    print(f"Loaded {len(self.mimicked_samples)} individual mimicked samples.")
                    return True
                else:
                    print(f"No mimicked content found in {self.input_dir}")
                    return False
        except Exception as e:
            print(f"Error loading mimicked content: {e}")
            return False
    
    def extract_content_features(self):
        """Extract key linguistic features from mimicked content"""
        if not self.mimicked_samples:
            if not self.load_mimicked_content():
                return {}
        
        # Collect all content texts
        content_texts = [sample.get('content', '') for sample in self.mimicked_samples]
        
        # Initialize feature extraction
        features = {
            'word_frequencies': Counter(),
            'phrase_patterns': Counter(),
            'sentiment_distribution': {},
            'topic_distribution': None,
            'conceptual_relationships': []
        }
        
        # Word frequencies (excluding common stopwords)
        vectorizer = CountVectorizer(stop_words='english', max_features=100)
        word_counts = vectorizer.fit_transform(content_texts)
        words = vectorizer.get_feature_names_out()
        word_freq = np.asarray(word_counts.sum(axis=0)).ravel()
        word_freq_dict = dict(zip(words, word_freq))
        features['word_frequencies'] = Counter(word_freq_dict)
        
        # Common phrases (n-grams)
        phrase_vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english', max_features=50)
        phrase_counts = phrase_vectorizer.fit_transform(content_texts)
        phrases = phrase_vectorizer.get_feature_names_out()
        phrase_freq = np.asarray(phrase_counts.sum(axis=0)).ravel()
        phrase_freq_dict = dict(zip(phrases, phrase_freq))
        features['phrase_patterns'] = Counter(phrase_freq_dict)
        
        # Topic modeling (Latent Dirichlet Allocation)
        tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(content_texts)
        
        # Find optimal number of topics
        num_topics = min(5, len(content_texts))
        
        try:
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(tfidf)
            
            # Extract topics
            feature_names = tfidf_vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                topics.append({
                    'id': topic_idx,
                    'top_words': top_words,
                    'weight': float(np.sum(topic))
                })
            
            features['topic_distribution'] = topics
        except Exception as e:
            print(f"Error in topic modeling: {e}")
        
        # Extract conceptual relationships
        for sample in self.mimicked_samples:
            # Extract entities and their relationships from content
            content = sample.get('content', '')
            # Simple pattern matching for relationship extraction
            # Looking for patterns like "X is Y" or "X does Y"
            relationship_patterns = [
                (r'(\b\w+\b)\s+is\s+(\b\w+\b)', 'is'),
                (r'(\b\w+\b)\s+are\s+(\b\w+\b)', 'are'),
                (r'(\b\w+\b)\s+has\s+(\b\w+\b)', 'has'),
                (r'(\b\w+\b)\s+contains\s+(\b\w+\b)', 'contains')
            ]
            
            for pattern, relation_type in relationship_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    if match and len(match.groups()) >= 2:
                        source = match.group(1)
                        target = match.group(2)
                        features['conceptual_relationships'].append({
                            'source': source,
                            'relation': relation_type,
                            'target': target,
                            'sample_id': sample.get('id')
                        })
        
        return features
    
    def analyze_path_patterns(self):
        """Analyze patterns in the walk paths from mimicked content"""
        if not self.mimicked_samples:
            if not self.load_mimicked_content():
                return {}
                
        path_analytics = {
            'common_nodes': Counter(),
            'transitions': Counter(),
            'path_lengths': [],
            'connection_density': {}
        }
        
        # Extract all paths
        all_paths = []
        for sample in self.mimicked_samples:
            paths = sample.get('paths', [])
            all_paths.extend(paths)
        
        # Count node occurrences
        for path in all_paths:
            for node in path:
                path_analytics['common_nodes'][node] += 1
                
        # Count transitions
        transitions_dict = {}
        for path in all_paths:
            for i in range(len(path) - 1):
                transition = (path[i], path[i+1])
                transition_key = f"{path[i]}_to_{path[i+1]}"
                if transition_key in transitions_dict:
                    transitions_dict[transition_key] += 1
                else:
                    transitions_dict[transition_key] = 1
                    
        path_analytics['transitions'] = Counter(transitions_dict)
        
        # Path length distribution
        path_analytics['path_lengths'] = [len(path) for path in all_paths]
        
        # Connection density (how interconnected the concepts are)
        G = nx.DiGraph()
        for path in all_paths:
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i+1])
        
        if len(G.nodes()) > 0:
            path_analytics['connection_density'] = {
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G.to_undirected())
            }
        
        return path_analytics
    
    def evaluate_content_quality(self):
        """Evaluate the quality of mimicked content"""
        if not self.mimicked_samples:
            if not self.load_mimicked_content():
                return {}
        
        quality_metrics = {
            'coherence_scores': [],
            'complexity_metrics': [],
            'novelty_assessment': [],
            'overall_quality': {}
        }
        
        # Collect all content texts
        content_texts = [sample.get('content', '') for sample in self.mimicked_samples]
        
        # Measure coherence based on sentence flow
        for i, content in enumerate(content_texts):
            sentences = re.split(r'[.!?]', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 3:
                coherence_score = 0.5  # Neutral score for very short content
            else:
                # Count transition words as a proxy for coherence
                transition_words = ['however', 'therefore', 'consequently', 'furthermore', 
                                   'moreover', 'thus', 'hence', 'accordingly', 'besides',
                                   'additionally', 'nonetheless', 'meanwhile', 'subsequently',
                                   'in conclusion', 'in summary']
                
                transition_count = sum(1 for word in transition_words 
                                     if re.search(r'\b' + word + r'\b', content, re.IGNORECASE))
                
                # Normalize by content length
                coherence_score = min(1.0, transition_count / (len(content) / 500))
                
            quality_metrics['coherence_scores'].append({
                'sample_id': i + 1,
                'score': coherence_score
            })
        
        # Measure complexity based on sentence length and word length
        for i, content in enumerate(content_texts):
            sentences = re.split(r'[.!?]', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                complexity_score = 0
            else:
                # Average sentence length
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                
                # Average word length
                words = content.split()
                avg_word_length = np.mean([len(word) for word in words]) if words else 0
                
                # Combine into complexity score (normalized)
                complexity_score = (avg_sentence_length / 20 + avg_word_length / 8) / 2
            
            quality_metrics['complexity_metrics'].append({
                'sample_id': i + 1,
                'complexity': min(1.0, complexity_score),
                'avg_sentence_length': float(avg_sentence_length) if sentences else 0,
                'avg_word_length': float(avg_word_length) if words else 0
            })
        
        # Assess novelty by comparing samples to each other
        if len(content_texts) > 1:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(content_texts)
            
            # Compute pairwise similarities
            pairwise_similarity = cosine_similarity(tfidf_matrix)
            
            # For each sample, calculate average similarity to others (lower is more novel)
            for i in range(len(content_texts)):
                # Exclude self-similarity (diagonal)
                similarities = [pairwise_similarity[i, j] for j in range(len(content_texts)) if i != j]
                avg_similarity = np.mean(similarities) if similarities else 0
                novelty_score = 1 - avg_similarity  # Convert similarity to novelty
                
                quality_metrics['novelty_assessment'].append({
                    'sample_id': i + 1,
                    'novelty_score': float(novelty_score),
                    'avg_similarity_to_others': float(avg_similarity)
                })
        
        # Calculate overall quality metrics
        if quality_metrics['coherence_scores'] and quality_metrics['complexity_metrics'] and quality_metrics['novelty_assessment']:
            avg_coherence = np.mean([item['score'] for item in quality_metrics['coherence_scores']])
            avg_complexity = np.mean([item['complexity'] for item in quality_metrics['complexity_metrics']])
            avg_novelty = np.mean([item['novelty_score'] for item in quality_metrics['novelty_assessment']])
            
            # Combined quality score
            overall_quality = (avg_coherence + avg_complexity + avg_novelty) / 3
            
            quality_metrics['overall_quality'] = {
                'score': float(overall_quality),
                'avg_coherence': float(avg_coherence),
                'avg_complexity': float(avg_complexity),
                'avg_novelty': float(avg_novelty)
            }
        
        return quality_metrics
    
    def harvest(self):
        """Extract insights from mimicked content"""
        # Load content if not already loaded
        if not self.mimicked_samples:
            if not self.load_mimicked_content():
                return None
        
        print("Harvesting insights from mimicked content...")
        
        # Extract various types of insights
        content_features = self.extract_content_features()
        path_patterns = self.analyze_path_patterns()
        quality_assessment = self.evaluate_content_quality()
        
        # Combine all insights
        self.harvested_insights = {
            'content_features': content_features,
            'path_patterns': path_patterns,
            'quality_assessment': quality_assessment,
            'meta': {
                'sample_count': len(self.mimicked_samples),
                'harvested_at': time.time(),
                'source': self.input_dir
            }
        }
        
        # Create visualizations
        self.generate_visualizations()
        
        # Save insights to file
        self.save_insights()
        
        return self.harvested_insights
    
    def generate_visualizations(self):
        """Generate visualizations of the harvested insights"""
        insights = self.harvested_insights
        
        if not insights:
            print("No insights to visualize.")
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # 1. Word cloud from word frequencies
        word_freq = insights.get('content_features', {}).get('word_frequencies', {})
        if word_freq:
            plt.figure(figsize=(10, 8))
            wc = WordCloud(background_color="white", width=800, height=600, 
                          colormap='viridis', max_words=100)
            wc.generate_from_frequencies(word_freq)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title('Key Concepts in Mimicked Content')
            plt.savefig(os.path.join(self.viz_dir, 'word_cloud.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Topic distribution
        topics = insights.get('content_features', {}).get('topic_distribution', [])
        if topics:
            plt.figure(figsize=(12, 6))
            topic_names = [', '.join(topic['top_words'][:3]) for topic in topics]
            topic_weights = [topic['weight'] for topic in topics]
            
            # Normalize weights
            if sum(topic_weights) > 0:
                topic_weights = [w / sum(topic_weights) for w in topic_weights]
            
            plt.barh(topic_names, topic_weights, color='skyblue')
            plt.xlabel('Relative Weight')
            plt.title('Topic Distribution in Mimicked Content')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'topic_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Quality metrics comparison
        quality = insights.get('quality_assessment', {}).get('overall_quality', {})
        if quality:
            categories = ['Coherence', 'Complexity', 'Novelty', 'Overall']
            scores = [quality.get('avg_coherence', 0), 
                     quality.get('avg_complexity', 0),
                     quality.get('avg_novelty', 0),
                     quality.get('score', 0)]
            
            plt.figure(figsize=(8, 6))
            plt.bar(categories, scores, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
            plt.ylim(0, 1)
            plt.ylabel('Score')
            plt.title('Content Quality Assessment')
            plt.savefig(os.path.join(self.viz_dir, 'quality_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Concept relationships network
        relationships = insights.get('content_features', {}).get('conceptual_relationships', [])
        if relationships:
            G = nx.DiGraph()
            
            # Add edges for each relationship
            for rel in relationships:
                source = rel['source'].lower()
                target = rel['target'].lower()
                relation = rel['relation']
                
                G.add_edge(source, target, label=relation)
            
            # Filter to keep the graph manageable (max 20 nodes)
            if len(G.nodes()) > 20:
                # Keep nodes with highest degree
                degrees = dict(G.degree())
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
                top_node_names = [node for node, _ in top_nodes]
                G = nx.subgraph(G, top_node_names)
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, k=0.15, iterations=50)
            
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=10)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
            
            plt.title("Conceptual Relationships Network")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'concept_network.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {self.viz_dir}")
    
    def _ensure_json_serializable(self, obj):
        """Convert NumPy types and non-serializable objects to standard Python types for JSON serialization"""
        if isinstance(obj, dict):
            # Convert any non-string keys to strings (especially tuples)
            return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k: 
                    self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (float, np.float64, np.float32)):  # np.float removed
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._ensure_json_serializable(obj.tolist())
        elif isinstance(obj, tuple):
            # Convert tuple to string when it's likely to be used as a key
            if len(obj) == 2 and all(isinstance(x, str) for x in obj):
                return f"{obj[0]}_to_{obj[1]}"
            else:
                return tuple(self._ensure_json_serializable(item) for item in obj)
        elif hasattr(obj, 'item'):  # Handle any other numpy scalar types
            return obj.item()
        else:
            return obj

    def save_insights(self):
        """Save harvested insights to file"""
        if not self.harvested_insights:
            print("No insights to save.")
            return
            
        # Convert NumPy types to standard Python types
        serializable_insights = self._ensure_json_serializable(self.harvested_insights)
        
        output_file = os.path.join(self.output_dir, 'harvested_insights.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_insights, f, indent=2)
            
        print(f"Harvested insights saved to {output_file}")
        
        # Save a summary file with key findings
        summary = self.generate_summary()
        summary_file = os.path.join(self.output_dir, 'insights_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        print(f"Summary saved to {summary_file}")
    
    def generate_summary(self):
        """Generate a textual summary of key insights"""
        insights = self.harvested_insights
        if not insights:
            return "No insights available."
            
        summary_parts = ["# HARVESTED INSIGHTS SUMMARY", ""]
        
        # Summary of content features
        summary_parts.append("## Content Analysis")
        
        # Top words
        word_freq = insights.get('content_features', {}).get('word_frequencies', {})
        if word_freq:
            top_words = [word for word, _ in word_freq.most_common(10)]
            summary_parts.append(f"Top concepts: {', '.join(top_words)}")
        
        # Topics
        topics = insights.get('content_features', {}).get('topic_distribution', [])
        if topics:
            summary_parts.append("\nIdentified topics:")
            for i, topic in enumerate(topics):
                topic_terms = ', '.join(topic['top_words'][:5])
                summary_parts.append(f"- Topic {i+1}: {topic_terms}")
        
        # Path patterns
        path_patterns = insights.get('path_patterns', {})
        if path_patterns:
            summary_parts.append("\n## Path Analysis")
            
            common_nodes = path_patterns.get('common_nodes', {})
            if common_nodes:
                top_nodes = [node for node, _ in common_nodes.most_common(5)]
                summary_parts.append(f"Central concepts: {', '.join(top_nodes)}")
            
            connection_density = path_patterns.get('connection_density', {})
            if connection_density:
                density = connection_density.get('density', 0)
                clustering = connection_density.get('average_clustering', 0)
                summary_parts.append(f"Network density: {density:.3f}")
                summary_parts.append(f"Average clustering: {clustering:.3f}")
        
        # Quality assessment
        quality = insights.get('quality_assessment', {}).get('overall_quality', {})
        if quality:
            summary_parts.append("\n## Quality Assessment")
            summary_parts.append(f"Overall quality score: {quality.get('score', 0):.2f}/1.00")
            summary_parts.append(f"Coherence: {quality.get('avg_coherence', 0):.2f}/1.00")
            summary_parts.append(f"Complexity: {quality.get('avg_complexity', 0):.2f}/1.00")
            summary_parts.append(f"Novelty: {quality.get('avg_novelty', 0):.2f}/1.00")
        
        # Recommendations
        summary_parts.append("\n## Recommendations")
        
        # Add recommendations based on insights
        if quality and quality.get('avg_coherence', 0) < 0.5:
            summary_parts.append("- Improve coherence by strengthening connections between concepts")
            
        if path_patterns and path_patterns.get('connection_density', {}).get('density', 1) < 0.2:
            summary_parts.append("- Increase connectivity between concepts to create a more robust knowledge structure")
            
        if quality and quality.get('avg_novelty', 0) < 0.5:
            summary_parts.append("- Introduce more diverse concepts to increase novelty")
            
        # Add generic recommendations if specific ones weren't added
        if len(summary_parts) < 3:
            summary_parts.append("- Continue to evolve the knowledge graph with new information sources")
            summary_parts.append("- Focus on strengthening relationships between key concepts")
            summary_parts.append("- Consider expanding into adjacent domains to increase system robustness")
        
        return "\n".join(summary_parts)

import time

if __name__ == "__main__":
    # Example usage
    harvester = Harvester(input_dir='mimicked_content', output_dir='harvested_insights')
    harvested_data = harvester.harvest()