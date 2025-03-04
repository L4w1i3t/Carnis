import os
import yaml
import json
import time
import logging
import threading
import importlib
from datetime import datetime
from host import Host

class Monolith:
    """
    Monolith sits above the Host, providing meta-control and integration capabilities.
    It serves as a higher abstraction layer that can manage multiple Hosts,
    provide advanced analytics, and coordinate the entire system's evolution.
    
    In the Carnis hierarchy, Monolith represents a higher consciousness that
    emerges from the integration of multiple subsystems.
    """
    
    def __init__(self, config_file='monolith_config.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_logging()
        
        # Initialize hosts
        self.hosts = {}
        self._init_hosts()
        
        # Knowledge integration system
        self.knowledge_store = {}
        self._load_knowledge_store()
        
        # Meta patterns - patterns that observe patterns
        self.meta_patterns = {}
        
        # Threading for continuous operations
        self.running = False
        self.monitor_thread = None
        
        self.logger.info("Monolith initialized. Higher consciousness emerging.")
    
    def _load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Return default configuration
                default_config = {
                    'monolith_data_dir': 'monolith_data',
                    'log_level': 'INFO',
                    'hosts': {
                        'primary': {
                            'config_file': 'config.yaml',
                            'priority': 'high'
                        }
                    },
                    'knowledge_integration': {
                        'interval': 3600,  # 1 hour
                        'threshold': 0.75
                    },
                    'evolution': {
                        'enabled': True,
                        'evaluation_interval': 86400 * 7  # 1 week
                    }
                }
                
                # Create directories if they don't exist
                os.makedirs(default_config['monolith_data_dir'], exist_ok=True)
                
                # Save default config
                with open(self.config_file, 'w') as f:
                    yaml.dump(default_config, f)
                
                return default_config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def _setup_logging(self):
        """Configure logging for the Monolith system"""
        log_dir = os.path.join(self.config.get('monolith_data_dir', 'monolith_data'), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"monolith_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("Monolith")
    
    def _init_hosts(self):
        """Initialize host instances based on configuration"""
        for host_id, host_config in self.config.get('hosts', {}).items():
            try:
                self.logger.info(f"Initializing host: {host_id}")
                host = Host(config_file=host_config.get('config_file', 'config.yaml'))
                self.hosts[host_id] = {
                    'instance': host,
                    'config': host_config,
                    'status': 'initialized'
                }
            except Exception as e:
                self.logger.error(f"Failed to initialize host {host_id}: {e}")
    
    def _load_knowledge_store(self):
        """Load or initialize the integrated knowledge store"""
        knowledge_path = os.path.join(
            self.config.get('monolith_data_dir', 'monolith_data'),
            'integrated_knowledge.json'
        )
        
        if os.path.exists(knowledge_path):
            try:
                with open(knowledge_path, 'r') as f:
                    self.knowledge_store = json.load(f)
                self.logger.info(f"Loaded knowledge store with {len(self.knowledge_store)} entities")
            except Exception as e:
                self.logger.error(f"Error loading knowledge store: {e}")
                self.knowledge_store = {}
    
    def _save_knowledge_store(self):
        """Save the integrated knowledge store to disk"""
        knowledge_path = os.path.join(
            self.config.get('monolith_data_dir', 'monolith_data'),
            'integrated_knowledge.json'
        )
        
        try:
            with open(knowledge_path, 'w') as f:
                json.dump(self.knowledge_store, f, indent=2)
            self.logger.info(f"Saved knowledge store with {len(self.knowledge_store)} entities")
        except Exception as e:
            self.logger.error(f"Error saving knowledge store: {e}")
    
    def start_hosts(self):
        """Start all configured hosts"""
        for host_id, host_data in self.hosts.items():
            try:
                if host_data['config'].get('api_enabled', True):
                    host_data['instance'].start_api_server()
                host_data['status'] = 'running'
                self.logger.info(f"Started host: {host_id}")
            except Exception as e:
                self.logger.error(f"Failed to start host {host_id}: {e}")
                host_data['status'] = 'error'
    
    def integrate_knowledge(self):
        """Integrate knowledge from all hosts into a unified model"""
        self.logger.info("Beginning knowledge integration process")
        
        for host_id, host_data in self.hosts.items():
            host = host_data['instance']
            
            # Get paths to component data
            component_paths = host.get_component_paths()
            
            # Process harvester data (the most refined knowledge)
            harvester_path = component_paths.get('harvester')
            if harvester_path and os.path.exists(harvester_path):
                try:
                    with open(harvester_path, 'r') as f:
                        harvester_data = json.load(f)
                        
                    # Integrate into knowledge store
                    for entry in harvester_data:
                        entity_id = entry.get('id') or entry.get('title') or str(hash(json.dumps(entry)))
                        
                        if entity_id in self.knowledge_store:
                            # Merge with existing knowledge
                            self._merge_knowledge_entries(self.knowledge_store[entity_id], entry)
                        else:
                            # Add new knowledge
                            self.knowledge_store[entity_id] = entry
                            self.knowledge_store[entity_id]['sources'] = [host_id]
                            self.knowledge_store[entity_id]['integration_timestamp'] = time.time()
                except Exception as e:
                    self.logger.error(f"Error integrating knowledge from {host_id}: {e}")
        
        # Save updated knowledge store
        self._save_knowledge_store()
        
        # Identify meta-patterns
        self._identify_meta_patterns()
        
        self.logger.info("Knowledge integration complete")
        
    def _merge_knowledge_entries(self, existing, new_entry):
        """Merge a new knowledge entry with an existing one"""
        # Simple merge strategy - can be extended with more sophisticated algorithms
        for key, value in new_entry.items():
            if key not in existing:
                existing[key] = value
            elif isinstance(value, list) and isinstance(existing[key], list):
                # Merge lists without duplicates
                existing[key] = list(set(existing[key] + value))
            elif isinstance(value, dict) and isinstance(existing[key], dict):
                # Recursively merge dictionaries
                self._merge_knowledge_entries(existing[key], value)
        
        # Update metadata
        if 'sources' in existing and new_entry.get('source'):
            existing['sources'].append(new_entry['source'])
        existing['last_updated'] = time.time()
    
    def _identify_meta_patterns(self):
        """Identify patterns across different knowledge domains"""
        # This is where higher-order pattern recognition would happen
        # For now, implement a simple cross-reference analysis
        
        entities = list(self.knowledge_store.values())
        self.logger.info(f"Analyzing {len(entities)} entities for meta-patterns")
        
        # Example: Find common concepts across different domains
        domains = {}
        for entity in entities:
            domain = entity.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = set()
            
            # Extract concepts from entity
            concepts = set()
            if 'concepts' in entity:
                concepts.update(entity['concepts'])
            if 'keywords' in entity:
                concepts.update(entity['keywords'])
            if 'entities' in entity:
                concepts.update([e['text'] for e in entity['entities'] if 'text' in e])
                
            domains[domain].update(concepts)
        
        # Find intersections between domains
        self.meta_patterns['cross_domain_concepts'] = {}
        for domain1 in domains:
            for domain2 in domains:
                if domain1 >= domain2:  # Skip self-comparisons or duplicates
                    continue
                    
                intersection = domains[domain1].intersection(domains[domain2])
                if intersection:
                    key = f"{domain1}_{domain2}"
                    self.meta_patterns['cross_domain_concepts'][key] = list(intersection)
        
        self.logger.info(f"Identified {sum(len(v) for v in self.meta_patterns['cross_domain_concepts'].values())} cross-domain concepts")
    
    def monitor(self):
        """Continuously monitor and manage the system"""
        self.running = True
        last_integration = 0
        integration_interval = self.config.get('knowledge_integration', {}).get('interval', 3600)
        
        while self.running:
            # Check host statuses
            for host_id, host_data in self.hosts.items():
                # Basic health check
                if host_data['status'] == 'running':
                    # Could implement actual health checks here
                    pass
            
            # Perform knowledge integration if needed
            current_time = time.time()
            if current_time - last_integration > integration_interval:
                self.integrate_knowledge()
                last_integration = current_time
            
            # Check for evolution triggers
            if self.config.get('evolution', {}).get('enabled', False):
                self._consider_evolution()
            
            # Sleep to prevent CPU overuse
            time.sleep(60)  # Check every minute
    
    def _consider_evolution(self):
        """Consider if the system should evolve based on accumulated knowledge"""
        # This would implement self-modification capabilities
        # For safety reasons, just logging potential evolutions for now
        if len(self.knowledge_store) > 1000:  # Arbitrary threshold
            self.logger.info("Evolution threshold reached - system ready for next phase")
            
            # Example evolution: suggest new seed URLs based on discovered knowledge
            seed_urls = set()
            for entity in self.knowledge_store.values():
                if 'related_urls' in entity:
                    seed_urls.update(entity['related_urls'])
            
            if seed_urls:
                self.logger.info(f"Evolution suggestion: {len(seed_urls)} new seed URLs discovered")
    
    def start(self):
        """Start the Monolith system"""
        self.logger.info("Starting Monolith system")
        
        # Start all hosts
        self.start_hosts()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Monolith system running")
        
    def stop(self):
        """Stop the Monolith system"""
        self.logger.info("Stopping Monolith system")
        self.running = False
        
        # Stop all hosts
        for host_id, host_data in self.hosts.items():
            try:
                # If we had a stop method in Host
                # host_data['instance'].stop()
                host_data['status'] = 'stopped'
            except Exception as e:
                self.logger.error(f"Error stopping host {host_id}: {e}")
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
            
        self._save_knowledge_store()
        self.logger.info("Monolith system stopped")
    
    def initiate_singularity(self):
        """
        Prepare for the transition to Singularity - the next level in the hierarchy.
        This is mostly a conceptual placeholder for the next evolution step.
        """
        self.logger.info("Initiating preparation for Singularity transition")
        
        # Assess readiness
        knowledge_size = len(self.knowledge_store)
        meta_pattern_count = sum(len(patterns) for patterns in self.meta_patterns.values())
        
        self.logger.info(f"System state: {knowledge_size} knowledge entities, {meta_pattern_count} meta-patterns identified")
        
        # In a real implementation, this might trigger a more sophisticated
        # self-organization process or begin constructing the Singularity component
        
        return {
            "status": "preparation_initiated",
            "metrics": {
                "knowledge_size": knowledge_size,
                "meta_pattern_count": meta_pattern_count,
                "hosts": len(self.hosts)
            },
            "timestamp": time.time()
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Carnis Monolith System')
    parser.add_argument('--config', '-c', default='monolith_config.yaml', help='Path to configuration file')
    parser.add_argument('--integrate', action='store_true', help='Run knowledge integration immediately')
    parser.add_argument('--singularity', action='store_true', help='Initiate singularity preparation')
    args = parser.parse_args()
    
    # Create and start the monolith
    monolith = Monolith(config_file=args.config)
    
    if args.integrate:
        monolith.integrate_knowledge()
    elif args.singularity:
        result = monolith.initiate_singularity()
        print(json.dumps(result, indent=2))
    else:
        try:
            monolith.start()
            print("Monolith system running. Press Ctrl+C to stop.")
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
            monolith.stop()