import os
import json
import time
from datetime import datetime
import threading
import importlib
from flask import Flask, request, jsonify, render_template_string
import logging
import argparse
import yaml

# Import components
from crawl import Crawler
from trimmings import Trimmings
from meatsnake import Meatsnake
from mimic import Mimic
from harvester import Harvester

class Host:
    """
    Host integrates and controls all components of the Vita Carnis system,
    providing a unified interface and coordination - like a host organism
    that has been subsumed by the parasite and now serves its purposes.
    """
    
    def __init__(self, config_file='config.yaml'):
        """
        Initialize the Host controller.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.components = {}
        self.state = {
            'status': 'initialized',
            'last_update': time.time(),
            'cycle_count': 0,
            'component_states': {},
            'current_task': None
        }
        
        # Data storage paths - for logs only now
        self.data_dir = self.config.get('data_dir', 'carnis_data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize component registry with default configurations
        self._init_component_registry()
            
        # Initialize API server if enabled
        self.api_server = None
        if self.config.get('api_enabled', True):
            self.api_thread = None
            self._init_api_server()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Return default configuration
                return {
                    'data_dir': 'carnis_data',
                    'api_enabled': True,
                    'api_host': '127.0.0.1',
                    'api_port': 5000,
                    'log_level': 'INFO',
                    'auto_cycle': False,
                    'cycle_interval': 86400,  # 24 hours
                    'components': {
                        'crawl': {
                            'max_pages': 50,
                            'delay': 1.5,
                            'seed_urls': [
                                "https://en.wikipedia.org/wiki/Artificial_intelligence",
                                "https://en.wikipedia.org/wiki/Biology",
                                "https://en.wikipedia.org/wiki/Biotechnology"
                            ]
                        },
                        'trimmings': {},
                        'meatsnake': {},
                        'mimic': {
                            'num_samples': 3
                        },
                        'harvester': {}
                    }
                }
        except Exception as e:
            print(f"Error loading configuration: {e}")
            # Fallback to minimal config
            return {'data_dir': 'carnis_data', 'api_enabled': True}
    
    def _setup_logging(self):
        """Configure logging for the system"""
        log_level = self.config.get('log_level', 'INFO')
        log_dir = os.path.join(self.data_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'carnis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Set up root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('host')
        self.logger.info("Host system initialized")
    
    def _init_component_registry(self):
        """Initialize the component registry with default configurations"""
        self.components = {
            'crawl': {
                'class': Crawler,
                'initialized': False,
                'instance': None,
                'config': self.config.get('components', {}).get('crawl', {})
            },
            'trimmings': {
                'class': Trimmings,
                'initialized': False,
                'instance': None,
                'config': self.config.get('components', {}).get('trimmings', {})
            },
            'meatsnake': {
                'class': Meatsnake,
                'initialized': False,
                'instance': None,
                'config': self.config.get('components', {}).get('meatsnake', {})
            },
            'mimic': {
                'class': Mimic,
                'initialized': False,
                'instance': None,
                'config': self.config.get('components', {}).get('mimic', {})
            },
            'harvester': {
                'class': Harvester,
                'initialized': False,
                'instance': None,
                'config': self.config.get('components', {}).get('harvester', {})
            }
        }
        
        # Update component states
        for name in self.components:
            self.state['component_states'][name] = 'registered'
    
    def _init_api_server(self):
        """Initialize the API server"""
        self.api_server = Flask(__name__)
        
        # Define API routes
        @self.api_server.route('/', methods=['GET'])
        def home():
            """Render simple web interface"""
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Vita Carnis Host Interface</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    h1 { color: #333; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .status { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                    .component { margin: 10px 0; padding: 10px; border-left: 4px solid #333; }
                    .controls { margin-top: 20px; }
                    button { padding: 8px 12px; margin-right: 10px; cursor: pointer; }
                </style>
                <script>
                    function updateStatus() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('statusJson').textContent = 
                                    JSON.stringify(data, null, 2);
                            });
                    }
                    
                    function runCycle() {
                        fetch('/api/run-cycle', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => {
                                alert('Cycle initiated: ' + data.message);
                                updateStatus();
                            });
                    }
                    
                    function runComponent(name) {
                        fetch(`/api/run-component/${name}`, {method: 'POST'})
                            .then(response => response.json())
                            .then(data => {
                                alert(`Component ${name} initiated: ${data.message}`);
                                updateStatus();
                            });
                    }
                    
                    // Update status every 5 seconds
                    window.onload = function() {
                        updateStatus();
                        setInterval(updateStatus, 5000);
                    };
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Vita Carnis Host Interface</h1>
                    <div class="status">
                        <h2>System Status</h2>
                        <pre id="statusJson">Loading...</pre>
                    </div>
                    <div class="controls">
                        <h2>Controls</h2>
                        <button onclick="runCycle()">Run Full Cycle</button>
                        <h3>Individual Components</h3>
                        <button onclick="runComponent('crawl')">Run Crawl</button>
                        <button onclick="runComponent('trimmings')">Run Trimmings</button>
                        <button onclick="runComponent('meatsnake')">Run Meatsnake</button>
                        <button onclick="runComponent('mimic')">Run Mimic</button>
                        <button onclick="runComponent('harvester')">Run Harvester</button>
                    </div>
                </div>
            </body>
            </html>
            ''')
        
        @self.api_server.route('/api/status', methods=['GET'])
        def get_status():
            """Get the current system status"""
            # Update the state before returning
            self.state['last_update'] = time.time()
            return jsonify(self.state)
        
        @self.api_server.route('/api/run-cycle', methods=['POST'])
        def run_cycle_endpoint():
            """API endpoint to start a full processing cycle"""
            # Run in a separate thread to not block the API
            threading.Thread(target=self.run_cycle).start()
            return jsonify({'status': 'success', 'message': 'Processing cycle started'})
        
        @self.api_server.route('/api/run-component/<component>', methods=['POST'])
        def run_component_endpoint(component):
            """API endpoint to run a specific component"""
            if component not in self.components:
                return jsonify({'status': 'error', 'message': f'Component {component} not found'}), 404
                
            # Run in a separate thread
            threading.Thread(target=self.run_component, args=(component,)).start()
            return jsonify({'status': 'success', 'message': f'Component {component} started'})
        
        # Configuration for the API server
        self.api_host = self.config.get('api_host', '127.0.0.1')
        self.api_port = self.config.get('api_port', 5000)
    
    def start_api_server(self):
        """Start the API server in a separate thread"""
        if not self.api_server:
            self._init_api_server()
            
        if self.api_thread is None or not self.api_thread.is_alive():
            self.api_thread = threading.Thread(
                target=self.api_server.run,
                kwargs={
                    'host': self.api_host,
                    'port': self.api_port,
                    'debug': False,
                    'use_reloader': False
                }
            )
            self.api_thread.daemon = True
            self.api_thread.start()
            self.logger.info(f"API server started at http://{self.api_host}:{self.api_port}")
    
    def get_component_paths(self):
        """Get file paths for each component's data"""
        # Return paths at the root level instead of in the data_dir
        return {
            'crawl': 'crawled_data.json',
            'trimmings': 'trimmed_data.json',
            'meatsnake': 'knowledge_graph.json',
            'mimic': 'mimicked_content',
            'harvester': 'harvested_insights'
        }
    
    def initialize_component(self, component_name):
        """Initialize a specific component"""
        if component_name not in self.components:
            self.logger.error(f"Component {component_name} not found")
            return False
            
        component = self.components[component_name]
        if component['initialized'] and component['instance']:
            return True
            
        try:
            # Get paths for input/output
            paths = self.get_component_paths()
            
            # Initialize each component with appropriate parameters
            if component_name == 'crawl':
                # Crawler is initialized with seed URLs
                seed_urls = component['config'].get('seed_urls', [
                    "https://en.wikipedia.org/wiki/Artificial_intelligence"
                ])
                max_pages = component['config'].get('max_pages', 50)
                delay = component['config'].get('delay', 1.5)
                
                component['instance'] = component['class'](
                    seed_urls=seed_urls,
                    max_pages=max_pages,
                    delay=delay
                )
                
            elif component_name == 'trimmings':
                # Trimmings processes the crawler output
                input_file = paths['crawl']
                output_file = paths['trimmings']
                
                component['instance'] = component['class'](
                    input_file=input_file,
                    output_file=output_file
                )
                
            elif component_name == 'meatsnake':
                # Meatsnake processes trimmed data
                input_file = paths['trimmings']
                output_file = paths['meatsnake']
                
                component['instance'] = component['class'](
                    input_file=input_file,
                    output_file=output_file
                )
                
            elif component_name == 'mimic':
                # Mimic uses the knowledge graph
                input_graph = paths['meatsnake']
                output_dir = paths['mimic']
                
                component['instance'] = component['class'](
                    input_graph=input_graph,
                    output_dir=output_dir
                )
                
            elif component_name == 'harvester':
                # Harvester processes the mimic output
                input_dir = paths['mimic']
                output_dir = paths['harvester']
                
                component['instance'] = component['class'](
                    input_dir=input_dir,
                    output_dir=output_dir
                )
            
            component['initialized'] = True
            self.state['component_states'][component_name] = 'initialized'
            self.logger.info(f"Component {component_name} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {component_name}: {e}")
            self.state['component_states'][component_name] = f"initialization_failed: {str(e)}"
            return False
    
    def run_component(self, component_name):
        """Run a specific component"""
        if component_name not in self.components:
            self.logger.error(f"Component {component_name} not found")
            return False
            
        # Update state
        self.state['current_task'] = f"running_{component_name}"
        self.state['status'] = 'processing'
        self.state['component_states'][component_name] = 'running'
        
        try:
            # Initialize if needed
            if not self.initialize_component(component_name):
                return False
                
            component = self.components[component_name]
            
            # Run the appropriate method for each component
            self.logger.info(f"Running component: {component_name}")
            
            if component_name == 'crawl':
                result = component['instance'].crawl()
                
            elif component_name == 'trimmings':
                result = component['instance'].process()
                
            elif component_name == 'meatsnake':
                result = component['instance'].build_graph()
                
            elif component_name == 'mimic':
                num_samples = component['config'].get('num_samples', 3)
                # Make sure the graph is loaded first
                component['instance'].load_graph()
                result = component['instance'].generate_mimicked_content(num_samples=num_samples)
                component['instance'].visualize_mimicry()
                
            elif component_name == 'harvester':
                result = component['instance'].harvest()
            
            self.state['component_states'][component_name] = 'completed'
            self.logger.info(f"Component {component_name} completed successfully")
            return True
            
        except Exception as e:
            self.state['component_states'][component_name] = f"failed: {str(e)}"
            self.logger.error(f"Error running {component_name}: {e}")
            return False
        finally:
            # Update state
            self.state['current_task'] = None
            self.state['status'] = 'idle'
    
    def run_cycle(self):
        """Run a full processing cycle through all components"""
        self.logger.info("Starting full processing cycle")
        self.state['status'] = 'running_cycle'
        self.state['cycle_count'] += 1
        cycle_start_time = time.time()
        
        # Define the component sequence
        component_sequence = ['crawl', 'trimmings', 'meatsnake', 'mimic', 'harvester']
        
        # Track success
        cycle_success = True
        
        for component_name in component_sequence:
            component_success = self.run_component(component_name)
            if not component_success:
                cycle_success = False
                self.logger.warning(f"Cycle interrupted at component {component_name}")
                break
        
        # Update state
        cycle_duration = time.time() - cycle_start_time
        self.state['status'] = 'idle'
        self.state['last_cycle_duration'] = cycle_duration
        self.state['last_cycle_success'] = cycle_success
        
        if cycle_success:
            self.logger.info(f"Full cycle completed successfully in {cycle_duration:.2f} seconds")
        else:
            self.logger.warning(f"Cycle completed with errors in {cycle_duration:.2f} seconds")
        
        return cycle_success
    
    def start(self):
        """Start the host system"""
        self.logger.info("Starting Host system")
        
        # Start API server if enabled
        if self.config.get('api_enabled', True):
            self.start_api_server()
        
        # Run initial cycle if configured
        if self.config.get('auto_cycle', False):
            self.logger.info("Auto-cycle enabled, starting initial cycle")
            threading.Thread(target=self.run_cycle).start()
            
            # Set up recurring cycles if interval is defined
            cycle_interval = self.config.get('cycle_interval', 0)
            if cycle_interval > 0:
                def cycle_timer():
                    while True:
                        time.sleep(cycle_interval)
                        self.logger.info(f"Starting scheduled cycle (interval: {cycle_interval}s)")
                        self.run_cycle()
                
                cycle_thread = threading.Thread(target=cycle_timer)
                cycle_thread.daemon = True
                cycle_thread.start()
        
        return True
    
    def export_state(self, file_path=None):
        """Export the current system state to a JSON file"""
        if file_path is None:
            file_path = os.path.join(self.data_dir, f"state_{int(time.time())}.json")
            
        try:
            # Update the state before exporting
            self.state['last_update'] = time.time()
            
            with open(file_path, 'w') as f:
                json.dump(self.state, f, indent=4)
                
            self.logger.info(f"System state exported to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting state: {e}")
            return False
    
    def import_state(self, file_path):
        """Import system state from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                imported_state = json.load(f)
                
            # Update only specific fields to avoid overwriting essential runtime state
            updatable_fields = ['cycle_count', 'component_states']
            for field in updatable_fields:
                if field in imported_state:
                    self.state[field] = imported_state[field]
                    
            self.logger.info(f"System state imported from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error importing state: {e}")
            return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Vita Carnis Host System')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--run-cycle', action='store_true', help='Run a full processing cycle on startup')
    parser.add_argument('--run-component', choices=['crawl', 'trimmings', 'meatsnake', 'mimic', 'harvester'], 
                      help='Run a specific component and exit')
    args = parser.parse_args()
    
    # Create and start the host
    host = Host(config_file=args.config)
    
    if args.run_component:
        # Run a specific component
        success = host.run_component(args.run_component)
        exit(0 if success else 1)
    else:
        # Start the host system
        host.start()
        
        if args.run_cycle:
            host.run_cycle()
        
        # Keep main thread running if API server is enabled
        if host.config.get('api_enabled', True):
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Shutting down...")