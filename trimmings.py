import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import string

class Trimmings:
    """
    Trimmings processes raw crawled data, removing unwanted parts and 
    preparing the content for further analysis - like trimming meat.
    """
    
    def __init__(self, input_file=None, output_file=None):
        """
        Initialize the Trimmings processor.
        
        Args:
            input_file (str): Path to the JSON file containing crawled data
            output_file (str): Path to save the processed data
        """
        import os
    
        # Set default paths within the carnis_data directory structure
        if input_file is None:
            input_file = os.path.join('carnis_data', 'crawl', 'crawled_data.json')
        
        if output_file is None:
            # Ensure the trimmings directory exists
            os.makedirs(os.path.join('carnis_data', 'trimmings'), exist_ok=True)
            output_file = os.path.join('carnis_data', 'trimmings', 'trimmed_data.json')

        self.input_file = input_file
        self.output_file = output_file
        self.raw_data = []
        self.trimmed_data = []
        
        print("Downloading and configuring required NLTK resources...")
        try:
            # Download necessary NLTK data with force=True to ensure it's fully downloaded
            nltk.download('punkt', quiet=False, force=True)
            nltk.download('stopwords', quiet=False, force=True)
            nltk.download('wordnet', quiet=False, force=True)
            
            # Explicitly verify and load the punkt tokenizer
            from nltk.data import load
            try:
                # Try to load the tokenizer models directly
                load('tokenizers/punkt/english.pickle')
                print("Successfully loaded English punkt tokenizer")
            except LookupError:
                print("Warning: Could not load English punkt tokenizer")
                # Try downloading punkt resources again with different path
                nltk.download('punkt', download_dir=nltk.data.path[0])
        except Exception as e:
            print(f"Error setting up NLTK resources: {e}")
            print("Attempting to continue with available resources...")
        
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError as e:
            print(f"Error loading NLTK components: {e}")
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                                'as', 'what', 'when', 'where', 'how', 'who', 'which', 'this',
                                'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                                'can', 'will', 'not', 'should', 'would', 'i', 'me', 'my', 'myself',
                                'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                                'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                                'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                                'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
                                'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                                'between', 'into', 'through', 'during', 'before', 'after', 'above',
                                'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                                'don', 'should', 'now'])
            
            # Simple lemmatizer function that just returns the word
            self.lemmatizer = type('DummyLemmatizer', (), {'lemmatize': lambda self, word, *args, **kwargs: word})()
    
    def load_data(self):
        """Load data from the input JSON file"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            print(f"Loaded {len(self.raw_data)} documents for processing.")
        except FileNotFoundError:
            print(f"Error: File {self.input_file} not found.")
            self.raw_data = []
        except json.JSONDecodeError:
            print(f"Error: File {self.input_file} contains invalid JSON.")
            self.raw_data = []
    
    def clean_text(self, text):
        """
        Clean and normalize text content.
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_important_content(self, text):
        """
        Extract important sentences and content from the text.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            dict: Important content extracted from text
        """
        # Custom direct sentence tokenization approach
        try:
            from nltk.tokenize import PunktSentenceTokenizer
            # Try to directly initialize the tokenizer
            tokenizer = PunktSentenceTokenizer()
            sentences = tokenizer.tokenize(text)
            print(f"Successfully tokenized text into {len(sentences)} sentences using PunktSentenceTokenizer")
        except Exception as e:
            print(f"Warning: Custom NLTK sentence tokenization failed ({str(e)}). Trying standard method...")
            try:
                # Standard tokenization approach
                sentences = sent_tokenize(text)
            except Exception as e:
                print(f"Warning: Standard NLTK sentence tokenization failed ({str(e)}). Using fallback method.")
                # Simple fallback sentence tokenizer
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not sentences:
            return {"summary": "", "keywords": []}
        
        # Score sentences based on word frequency
        word_freq = {}
        for sentence in sentences:
            try:
                for word in word_tokenize(sentence):
                    if word.lower() not in self.stop_words:
                        if word.lower() not in word_freq:
                            word_freq[word.lower()] = 1
                        else:
                            word_freq[word.lower()] += 1
            except LookupError:
                # Fallback word tokenization
                for word in re.findall(r'\w+', sentence.lower()):
                    if word not in self.stop_words:
                        if word not in word_freq:
                            word_freq[word] = 1
                        else:
                            word_freq[word] += 1
        
        # Calculate sentence scores
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_scores[i] = 0
            try:
                words = word_tokenize(sentence)
            except LookupError:
                words = re.findall(r'\w+', sentence.lower())
                
            for word in words:
                if word.lower() in word_freq:
                    sentence_scores[i] += word_freq[word.lower()]
        
        # Get top sentences (approximately 30% of the original content)
        top_n = max(3, int(len(sentences) * 0.3))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_sentences = [sentences[i] for i, _ in sorted(top_sentences, key=lambda x: x[0])]
        
        # Extract keywords (nouns and important terms)
        try:
            all_words = word_tokenize(text)
        except LookupError:
            all_words = re.findall(r'\w+', text.lower())
            
        filtered_words = []
        for word in all_words:
            try:
                lemmatized = self.lemmatizer.lemmatize(word.lower())
                if lemmatized not in self.stop_words and len(lemmatized) > 2:
                    filtered_words.append(lemmatized)
            except LookupError:
                if word.lower() not in self.stop_words and len(word) > 2:
                    filtered_words.append(word.lower())
        
        # Get keyword frequency
        keyword_freq = {}
        for word in filtered_words:
            if word in keyword_freq:
                keyword_freq[word] += 1
            else:
                keyword_freq[word] = 1
        
        # Select top keywords
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "summary": " ".join(top_sentences),
            "keywords": [kw for kw, _ in top_keywords]
        }
    
    def process(self):
        """Process all the raw data and generate trimmed output"""
        self.load_data()
        self.trimmed_data = []
        
        for idx, item in enumerate(self.raw_data):
            print(f"Processing document {idx+1}/{len(self.raw_data)}: {item['title']}")
            
            # Clean the content
            cleaned_text = self.clean_text(item['content'])
            
            # Extract important content
            extracted = self.extract_important_content(cleaned_text)
            
            # Create trimmed data object
            trimmed_item = {
                'url': item['url'],
                'title': item['title'],
                'summary': extracted['summary'],
                'keywords': extracted['keywords'],
                'timestamp': item['timestamp']
            }
            
            self.trimmed_data.append(trimmed_item)
        
        # Save processed data
        self.save_data()
        
        return self.trimmed_data
    
    def save_data(self):
        """Save processed data to output file"""
        import os
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Save the data
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.trimmed_data, f, indent=4)
        
        print(f"Trimming complete. Processed {len(self.trimmed_data)} documents.")
        print(f"Trimmed data saved to {self.output_file}")

if __name__ == "__main__":
    # Example usage with default paths that use the carnis_data directory structure
    trimmer = Trimmings()
    processed_data = trimmer.process()