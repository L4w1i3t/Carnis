import os
import json
import time
from datetime import datetime
import threading
import importlib
from flask import Flask, request, jsonify, render_template_string, send_from_directory
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
    Host integrates and controls all components of the Carnis system,
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
        
        # Data storage paths
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
                    loaded_config = yaml.safe_load(f)
                    # Handle the case where the file exists but is empty or invalid YAML
                    if loaded_config is None:
                        self.logger.warning(f"Config file {self.config_file} exists but is empty or invalid. Writing default configuration.")
                        # Write default config to the file
                        default_config = self._get_default_config()
                        with open(self.config_file, 'w') as f:
                            yaml.dump(default_config, f, default_flow_style=False)
                        return default_config
                    return loaded_config
            else:
                # Create default configuration file
                default_config = self._get_default_config()
                os.makedirs(os.path.dirname(os.path.abspath(self.config_file)), exist_ok=True)
                with open(self.config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                print(f"Created new configuration file at {self.config_file}")
                return default_config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            # Fallback to minimal config - make sure this ALWAYS returns a dictionary
            return {'data_dir': 'carnis_data', 'api_enabled': True}
            
    def _get_default_config(self):
        """Return the default configuration"""
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
            <!-- Replace the existing template_string in the home() function with this enhanced version -->
<!DOCTYPE html>
<html>
<head>
    <title>Carnis Host Interface</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tom-select@2.2.2/dist/css/tom-select.bootstrap5.min.css">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .card-header { font-weight: 600; }
        .component { transition: all 0.3s ease; }
        .component:hover { transform: translateY(-5px); }
        .component-running { border-left: 4px solid #17a2b8; animation: pulse 2s infinite; }
        .component-completed { border-left: 4px solid #28a745; }
        .component-failed { border-left: 4px solid #dc3545; }
        .status-badge { float: right; }
        .log-container { height: 300px; overflow-y: auto; background: #f8f9fa; border-radius: 4px; padding: 10px; font-family: monospace; }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(23, 162, 184, 0.5); }
            70% { box-shadow: 0 0 0 10px rgba(23, 162, 184, 0); }
            100% { box-shadow: 0 0 0 0 rgba(23, 162, 184, 0); }
        }
        .graph-viz { width: 100%; height: 400px; border: 1px solid #ddd; border-radius: 4px; }
        .tab-content { padding: 20px 0; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <header class="mb-4">
            <h1 class="display-4">Carnis <small class="text-muted">Host Interface</small></h1>
            <p class="lead">Monitoring and control panel for the Carnis system</p>
        </header>

        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="dashboard-tab" data-bs-toggle="tab" data-bs-target="#dashboard" type="button">Dashboard</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="logs-tab" data-bs-toggle="tab" data-bs-target="#logs" type="button">Logs</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="config-tab" data-bs-toggle="tab" data-bs-target="#config" type="button">Configuration</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="visualize-tab" data-bs-toggle="tab" data-bs-target="#visualize" type="button">Visualization</button>
            </li>
        </ul>

        <div class="tab-content" id="myTabContent">
            <!-- Dashboard Tab -->
            <div class="tab-pane fade show active" id="dashboard" role="tabpanel">
                <div class="dashboard">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            System Status
                        </div>
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h5>Current Status:</h5>
                                <span id="currentStatus" class="badge bg-secondary">Loading...</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h5>Current Task:</h5>
                                <span id="currentTask" class="badge bg-info">None</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h5>Cycle Count:</h5>
                                <span id="cycleCount">0</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h5>Last Update:</h5>
                                <span id="lastUpdate">-</span>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="card-header bg-success text-white">
                            Quick Controls
                        </div>
                        <div class="card-body">
                            <button id="runCycleBtn" class="btn btn-primary btn-lg mb-3 w-100">Run Full Cycle</button>
                            
                            <div class="mb-3">
                                <label for="componentSelect" class="form-label">Run Component:</label>
                                <select id="componentSelect" class="form-select">
                                    <option value="crawl">Crawl</option>
                                    <option value="trimmings">Trimmings</option>
                                    <option value="meatsnake">Meatsnake</option>
                                    <option value="mimic">Mimic</option>
                                    <option value="harvester">Harvester</option>
                                </select>
                            </div>
                            <button id="runComponentBtn" class="btn btn-success w-100">Run Selected Component</button>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-dark text-white">
                                Component Status
                            </div>
                            <div class="card-body">
                                <div id="componentStatusList"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Components Tab -->
            <div class="tab-pane fade" id="components" role="tabpanel">
                <div class="accordion" id="componentsAccordion">
                    <!-- Crawl -->
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#crawler">
                                Crawl <span id="crawlerStatus" class="badge bg-secondary ms-2">Unknown</span>
                            </button>
                        </h2>
                        <div id="crawler" class="accordion-collapse collapse show">
                            <div class="accordion-body">
                                <div class="mb-3">
                                    <label class="form-label">Seed URLs:</label>
                                    <select id="seedUrls" multiple placeholder="Add seed URLs..."></select>
                                </div>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="maxPages" class="form-label">Max Pages:</label>
                                            <input type="number" class="form-control" id="maxPages" value="50">
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="crawlDelay" class="form-label">Delay (seconds):</label>
                                            <input type="number" class="form-control" id="crawlDelay" step="0.1" value="1.5">
                                        </div>
                                    </div>
                                </div>
                                <button id="saveCrawlConfig" class="btn btn-primary">Save Configuration</button>
                                <button id="runCrawl" class="btn btn-success ms-2">Run Crawl</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Other components follow the same pattern -->
                </div>
            </div>

            <!-- Logs Tab -->
            <div class="tab-pane fade" id="logs" role="tabpanel">
                <div class="card">
                    <div class="card-header bg-dark text-white d-flex justify-content-between align-items-center">
                        <span>System Logs</span>
                        <button id="refreshLogs" class="btn btn-sm btn-outline-light">Refresh</button>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="logLevel" class="form-label">Log Level:</label>
                            <select id="logLevel" class="form-select form-select-sm w-auto">
                                <option value="all">All</option>
                                <option value="info">Info & Above</option>
                                <option value="warning">Warning & Above</option>
                                <option value="error">Errors Only</option>
                            </select>
                        </div>
                        <div class="log-container" id="logContent">Loading logs...</div>
                    </div>
                </div>
            </div>

            <!-- Configuration Tab -->
            <div class="tab-pane fade" id="config" role="tabpanel">
                <div class="card">
                    <div class="card-header bg-dark text-white">
                        System Configuration
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label for="configEditor" class="form-label">Configuration YAML:</label>
                            <textarea id="configEditor" class="form-control" style="height: 400px; font-family: monospace;"></textarea>
                        </div>
                        <button id="saveConfig" class="btn btn-primary">Save Configuration</button>
                        <button id="reloadConfig" class="btn btn-secondary ms-2">Reload</button>
                    </div>
                </div>
            </div>

            <!-- Visualization Tab -->
            <div class="tab-pane fade" id="visualize" role="tabpanel">
                <ul class="nav nav-pills mb-3">
                    <li class="nav-item">
                        <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#harvesterViz">Harvester</button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" data-bs-toggle="pill" data-bs-target="#mimicViz">Mimic Paths</button>
                    </li>
                </ul>
                <div class="tab-content">
                    <div class="tab-pane fade show active" id="harvesterViz">
                        <div class="card mb-4">
                            <div class="card-header">Harvested Insights</div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100">
                                            <div class="card-header">Word Cloud</div>
                                            <div class="card-body text-center">
                                                <img id="wordCloudImg" src="" alt="Word Cloud" class="img-fluid" style="max-height: 300px;">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100">
                                            <div class="card-header">Quality Metrics</div>
                                            <div class="card-body text-center">
                                                <img id="qualityMetricsImg" src="" alt="Quality Metrics" class="img-fluid" style="max-height: 300px;">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100">
                                            <div class="card-header">Topic Distribution</div>
                                            <div class="card-body text-center">
                                                <img id="topicDistributionImg" src="" alt="Topic Distribution" class="img-fluid" style="max-height: 300px;">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100">
                                            <div class="card-header">Concept Network</div>
                                            <div class="card-body text-center">
                                                <img id="conceptNetworkImg" src="" alt="Concept Network" class="img-fluid" style="max-height: 300px;">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="mt-3">
                                    <h5>Insights Summary</h5>
                                    <div id="insightsSummary" class="p-3 bg-light">Loading insights...</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="tab-pane fade" id="mimicViz">
                        <div class="card">
                            <div class="card-header">Mimic Paths Visualization</div>
                            <div class="card-body">
                                <div id="mimicSamples" class="mb-3"></div>
                                <div id="mimicPaths"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/tom-select@2.2.2/dist/js/tom-select.complete.min.js"></script>
    <script>
        // Initialize TomSelect for seed URLs
        let seedUrlsSelect = new TomSelect('#seedUrls', {
            create: true,
            createOnBlur: true,
            persist: false,
            placeholder: 'Add seed URLs...'
        });
        
        // Function to update status
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update dashboard elements
                    document.getElementById('currentStatus').textContent = data.status;
                    document.getElementById('currentStatus').className = 'badge ' + getStatusClass(data.status);
                    document.getElementById('currentTask').textContent = data.current_task || 'None';
                    document.getElementById('cycleCount').textContent = data.cycle_count;
                    document.getElementById('lastUpdate').textContent = new Date(data.last_update * 1000).toLocaleString();
                    
                    // Replace the existing component status list code
                    const componentList = document.getElementById('componentStatusList');
                    componentList.innerHTML = '';

                    // Define the hierarchical order
                    const componentOrder = ['crawl', 'trimmings', 'meatsnake', 'mimic', 'harvester'];
                        
                    // Loop through components in the defined order
                    for (const component of componentOrder) {
                        if (component in data.component_states) {
                            const status = data.component_states[component];
                            const statusClass = getComponentStatusClass(status);
                            const div = document.createElement('div');
                            div.className = `component p-3 mb-2 ${statusClass}`;
                            div.innerHTML = `
                                <h5>${capitalizeFirstLetter(component)} 
                                    <span class="status-badge badge ${getStatusBadgeClass(status)}">${status}</span>
                                </h5>
                                <div class="progress mb-2">
                                    <div class="progress-bar ${getProgressBarClass(status)}" 
                                        role="progressbar" 
                                        style="width: ${getProgressWidth(status)}%" 
                                        aria-valuenow="${getProgressWidth(status)}" 
                                        aria-valuemin="0" 
                                        aria-valuemax="100"></div>
                                </div>
                            `;
                            componentList.appendChild(div);
                            
                            // Also update the status on the components tab
                            const compStatus = document.getElementById(`${component}Status`);
                            if (compStatus) {
                                compStatus.textContent = status;
                                compStatus.className = 'badge ' + getStatusBadgeClass(status) + ' ms-2';
                            }
                        }
                    }
                    
                    // Populate config editor if it's empty
                    const configEditor = document.getElementById('configEditor');
                    if (configEditor.value === '') {
                        fetch('/api/config')
                            .then(response => response.text())
                            .then(config => {
                                configEditor.value = config;
                            });
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }
        
        // Helper functions for styling
        function getStatusClass(status) {
            switch (status) {
                case 'processing': return 'bg-info';
                case 'running_cycle': return 'bg-primary';
                case 'idle': return 'bg-success';
                case 'error': return 'bg-danger';
                default: return 'bg-secondary';
            }
        }
        
        function getComponentStatusClass(status) {
            if (status.includes('running')) {
                return 'component-running';
            } else if (status === 'completed') {
                return 'component-completed';
            } else if (status.includes('failed') || status.includes('error')) {
                return 'component-failed';
            } else {
                return '';
            }
        }
        
        function getStatusBadgeClass(status) {
            if (status.includes('running')) {
                return 'bg-info';
            } else if (status === 'completed') {
                return 'bg-success';
            } else if (status.includes('failed') || status.includes('error')) {
                return 'bg-danger';
            } else if (status === 'initialized') {
                return 'bg-primary';
            } else {
                return 'bg-secondary';
            }
        }
        
        function getProgressBarClass(status) {
            if (status.includes('running')) {
                return 'progress-bar-striped progress-bar-animated bg-info';
            } else if (status === 'completed') {
                return 'bg-success';
            } else if (status.includes('failed') || status.includes('error')) {
                return 'bg-danger';
            } else {
                return 'bg-secondary';
            }
        }
        
        function getProgressWidth(status) {
            if (status === 'registered') {
                return 0;
            } else if (status === 'initialized') {
                return 25;
            } else if (status.includes('running')) {
                return 75;
            } else if (status === 'completed') {
                return 100;
            } else if (status.includes('failed') || status.includes('error')) {
                return 100;
            } else {
                return 50;
            }
        }
        
        function capitalizeFirstLetter(string) {
            return string.charAt(0).toUpperCase() + string.slice(1);
        }
        
        // Event listeners
        document.getElementById('runCycleBtn').addEventListener('click', function() {
            fetch('/api/run-cycle', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(`Cycle initiated: ${data.message}`);
                    updateStatus();
                })
                .catch(error => {
                    console.error('Error starting cycle:', error);
                });
        });
        
        document.getElementById('runComponentBtn').addEventListener('click', function() {
            const component = document.getElementById('componentSelect').value;
            fetch(`/api/run-component/${component}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(`Component ${component} initiated: ${data.message}`);
                    updateStatus();
                })
                .catch(error => {
                    console.error(`Error starting component ${component}:`, error);
                });
        });
        
        document.getElementById('refreshLogs').addEventListener('click', function() {
            fetch('/api/logs')
                .then(response => response.text())
                .then(logs => {
                    document.getElementById('logContent').innerHTML = logs;
                })
                .catch(error => {
                    console.error('Error fetching logs:', error);
                });
        });
        
        document.getElementById('saveConfig').addEventListener('click', function() {
            const config = document.getElementById('configEditor').value;
            fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/yaml' },
                body: config
            })
                .then(response => response.json())
                .then(data => {
                    alert(`Configuration ${data.success ? 'saved' : 'failed to save'}: ${data.message}`);
                })
                .catch(error => {
                    console.error('Error saving config:', error);
                });
        });
        
        document.getElementById('reloadConfig').addEventListener('click', function() {
            fetch('/api/config')
                .then(response => response.text())
                .then(config => {
                    document.getElementById('configEditor').value = config;
                })
                .catch(error => {
                    console.error('Error loading config:', error);
                });
        });
        
        // Load mimic samples
        function loadMimicSamples() {
            fetch('/api/mimic-samples')
                .then(response => response.json())
                .then(samples => {
                    const samplesList = document.getElementById('mimicSamples');
                    samplesList.innerHTML = '';
                    
                    if (samples.length === 0) {
                        samplesList.innerHTML = '<div class="alert alert-info">No mimic samples available.</div>';
                        return;
                    }
                    
                    const select = document.createElement('select');
                    select.className = 'form-select mb-3';
                    select.id = 'mimicSampleSelect';
                    
                    samples.forEach((sample, index) => {
                        const option = document.createElement('option');
                        option.value = index;
                        option.textContent = `Sample ${sample.id}: ${sample.seed_concepts.join(', ').substring(0, 50)}...`;
                        select.appendChild(option);
                    });
                    
                    samplesList.appendChild(select);
                    select.addEventListener('change', function() {
                        displayMimicSample(samples[this.value]);
                    });
                    
                    if (samples.length > 0) {
                        displayMimicSample(samples[0]);
                    }
                })
                .catch(error => {
                    console.error('Error loading mimic samples:', error);
                });
        }
        
        function displayMimicSample(sample) {
            const pathsContainer = document.getElementById('mimicPaths');
            pathsContainer.innerHTML = '';
            
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <div class="card-header">
                    <h5>Sample ${sample.id}</h5>
                </div>
                <div class="card-body">
                    <h6>Seed Concepts:</h6>
                    <ul class="list-group mb-3">
                        ${sample.seed_concepts.map(concept => `<li class="list-group-item">${concept}</li>`).join('')}
                    </ul>
                    
                    <ul class="nav nav-tabs mb-3" id="sampleTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="details-tab" data-bs-toggle="tab" 
                                data-bs-target="#details-content" type="button">Details</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="paths-tab" data-bs-toggle="tab" 
                                data-bs-target="#paths-viz" type="button">Path Visualization</button>
                        </li>
                    </ul>
                    
                    <div class="tab-content">
                        <div class="tab-pane fade show active" id="details-content">
                            <h6>Prompt:</h6>
                            <div class="p-3 mb-3 bg-light">${sample.prompt}</div>
                            <h6>Generated Content:</h6>
                            <div class="p-3 mb-3 bg-light">${sample.content}</div>
                            <h6>Paths (text):</h6>
                            <ul class="list-group">
                                ${sample.paths.map(path => `
                                    <li class="list-group-item">
                                        ${path.join(' → ')}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                        <div class="tab-pane fade" id="paths-viz">
                            <div class="mb-3">
                                <div id="pathVisualization" style="width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 4px;"></div>
                            </div>
                            <div class="small text-muted">Note: The visualization shows the conceptual paths traversed during mimicry.</div>
                        </div>
                    </div>
                </div>
            `;
            
            pathsContainer.appendChild(card);
            
            // Initialize visualization when the paths tab is clicked
            document.getElementById('paths-tab').addEventListener('click', () => {
                renderPathVisualization(sample);
            });
        }

        function renderPathVisualization(sample) {
            // This function will create a visualization of the paths
            const container = document.getElementById('pathVisualization');
            
            // If we need D3 or another visualization library, we can load it dynamically
            if (!window.d3) {
                const script = document.createElement('script');
                script.src = 'https://d3js.org/d3.v7.min.js';
                script.onload = () => createVisualization(container, sample);
                document.head.appendChild(script);
            } else {
                createVisualization(container, sample);
            }
        }

                function createVisualization(container, sample) {
            // Clear any existing visualization
            container.innerHTML = '';
            
            // Basic SVG setup
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', 'translate(40,20)');
                
            // Create a unique list of all concepts in all paths
            const allNodes = new Set();
            sample.paths.forEach(path => {
                path.forEach(node => allNodes.add(node));
            });
            
            const nodes = Array.from(allNodes).map(name => ({ id: name, name }));
            
            // Create links from the paths
            const links = [];
            sample.paths.forEach(path => {
                for (let i = 0; i < path.length - 1; i++) {
                    links.push({
                        source: path[i],
                        target: path[i + 1],
                        value: 1
                    });
                }
            });
            
            // Combine duplicate links and increase their value
            const combinedLinks = {};
            links.forEach(link => {
                const key = `${link.source}|${link.target}`;
                if (combinedLinks[key]) {
                    combinedLinks[key].value += 1;
                } else {
                    combinedLinks[key] = link;
                }
            });
            
            const processedLinks = Object.values(combinedLinks);
            
            // Create a force-directed graph
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(processedLinks).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(50));
            
            // Draw the links
            const link = svg.append("g")
                .selectAll("line")
                .data(processedLinks)
                .enter().append("line")
                .attr("stroke-width", d => Math.sqrt(d.value) * 2)
                .attr("stroke", "#999")
                .attr("stroke-opacity", 0.6);
            
            // Create node groups
            const node = svg.append("g")
                .selectAll(".node")
                .data(nodes)
                .enter().append("g")
                .attr("class", "node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            // Add circles to nodes
            node.append("circle")
                .attr("r", 10)
                .attr("fill", (d) => {
                    if (sample.seed_concepts.includes(d.id)) {
                        return "#ff7f0e"; // orange for seed concepts
                    }
                    return "#1f77b4"; // default blue
                });
            
            // Add labels to nodes
            node.append("text")
                .attr("dx", 12)
                .attr("dy", ".35em")
                .text(d => d.name);
            
            // Update positions on simulation tick
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("transform", d => `translate(${d.x},${d.y})`);
            });
            
            // Drag functions for interactive movement
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            
            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            
            // Add legend
            const legend = svg.append("g")
                .attr("transform", "translate(20,20)");
                
            legend.append("circle")
                .attr("r", 6)
                .attr("fill", "#ff7f0e")
                .attr("cx", 0)
                .attr("cy", 0);
                
            legend.append("text")
                .attr("x", 15)
                .attr("y", 4)
                .text("Seed Concept");
                
            legend.append("circle")
                .attr("r", 6)
                .attr("fill", "#1f77b4")
                .attr("cx", 0)
                .attr("cy", 25);
                
            legend.append("text")
                .attr("x", 15)
                .attr("y", 29)
                .text("Path Node");
                
            // Add zoom capabilities
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on("zoom", (event) => {
                    svg.attr("transform", event.transform);
                });
                
            d3.select(container).select("svg")
                .call(zoom);
        }
                                    
        // Load harvester visualizations
        function loadHarvesterVisualizations() {
            // Load images
            document.getElementById('wordCloudImg').src = '/carnis_data/harvester/visualizations/word_cloud.png?t=' + new Date().getTime();
            document.getElementById('qualityMetricsImg').src = '/carnis_data/harvester/visualizations/quality_metrics.png?t=' + new Date().getTime();
            document.getElementById('topicDistributionImg').src = '/carnis_data/harvester/visualizations/topic_distribution.png?t=' + new Date().getTime();
            document.getElementById('conceptNetworkImg').src = '/carnis_data/harvester/visualizations/concept_network.png?t=' + new Date().getTime();
            
            // Load insights summary text
            fetch('/api/harvester-insights')
                .then(response => response.text())
                .then(text => {
                    document.getElementById('insightsSummary').innerHTML = formatInsightsSummary(text);
                })
                .catch(error => {
                    document.getElementById('insightsSummary').innerHTML = 
                        '<div class="alert alert-warning">Failed to load insights summary: ' + error.message + '</div>';
                });
        }
        
        // Format insights summary with better styling
        function formatInsightsSummary(text) {
            if (!text) return '<em>No insights available.</em>';
            
            // Split by line breaks and format as html
            const lines = text.split('\\n');
            let html = '';
            
            for (const line of lines) {
                if (line.trim() === '') continue;
                
                if (line.startsWith('-')) {
                    // This is a bullet point
                    html += '<li>' + line.substring(1).trim() + '</li>';
                } else if (line.startsWith('#')) {
                    // This is a header
                    html += '</ul><h6>' + line.substring(1).trim() + '</h6><ul>';
                } else {
                    // Regular paragraph
                    html += '<p>' + line + '</p>';
                }
            }
            
            // Wrap bullet points in ul tags
            if (html.includes('<li>')) {
                html = '<ul>' + html + '</ul>';
                // Clean up any empty ul tags
                html = html.replace('<ul></ul>', '');
            }
            
            return html;
        }
        
        // Initialize the interface
        document.addEventListener('DOMContentLoaded', function() {
            updateStatus();
            setInterval(updateStatus, 5000);
            
            // Initialize the components tab
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Populate crawl configuration
                    if (data.components && data.components.crawl) {
                        const crawlerConfig = data.components.crawl.config;
                        if (crawlerConfig.seed_urls) {
                            seedUrlsSelect.clear();
                            crawlerConfig.seed_urls.forEach(url => {
                                seedUrlsSelect.addOption({text: url, value: url});
                                seedUrlsSelect.addItem(url);
                            });
                        }
                        
                        document.getElementById('maxPages').value = crawlerConfig.max_pages || 50;
                        document.getElementById('crawlDelay').value = crawlerConfig.delay || 1.5;
                    }
                });
                
            // Load logs
            document.getElementById('refreshLogs').click();
            
            // Load visualizations when tabs are clicked
            document.getElementById('visualize-tab').addEventListener('click', function() {
                // Default to showing the harvester tab first
                document.querySelector('[data-bs-target="#harvesterViz"]').click();
                loadHarvesterVisualizations();
            });
            
            document.querySelector('[data-bs-target="#mimicViz"]').addEventListener('click', loadMimicSamples);
            document.querySelector('[data-bs-target="#harvesterViz"]').addEventListener('click', loadHarvesterVisualizations);
        });
    </script>
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
        
        @self.api_server.route('/api/config', methods=['GET'])
        def get_config():
            """Get the current configuration YAML"""
            try:
                with open(self.config_file, 'r') as f:
                    return f.read(), 200, {'Content-Type': 'application/yaml'}
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.api_server.route('/api/config', methods=['POST'])
        def update_config():
            """Update the configuration"""
            try:
                config_yaml = request.data.decode('utf-8')
                # Validate YAML format
                new_config = yaml.safe_load(config_yaml)
                
                # Write to file
                with open(self.config_file, 'w') as f:
                    f.write(config_yaml)
                    
                # Reload config
                self.config = self._load_config()
                self._init_component_registry()
                
                return jsonify({'success': True, 'message': 'Configuration updated successfully'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error updating configuration: {str(e)}'}), 400

        @self.api_server.route('/api/logs', methods=['GET'])
        def get_logs():
            """Get system logs"""
            log_level = request.args.get('level', 'all')
            log_file = os.path.join(self.data_dir, 'logs', sorted(os.listdir(os.path.join(self.data_dir, 'logs')))[-1])
            
            try:
                with open(log_file, 'r') as f:
                    logs = f.readlines()
                    
                # Filter logs by level if needed
                if log_level != 'all':
                    level_map = {
                        'debug': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        'info': ['INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        'warning': ['WARNING', 'ERROR', 'CRITICAL'],
                        'error': ['ERROR', 'CRITICAL']
                    }
                    logs = [log for log in logs if any(level in log for level in level_map.get(log_level.lower(), []))]
                
                # Format logs for HTML display
                formatted_logs = []
                for log in logs:
                    log_class = 'text-muted'
                    if 'ERROR' in log or 'CRITICAL' in log:
                        log_class = 'text-danger'
                    elif 'WARNING' in log:
                        log_class = 'text-warning'
                    elif 'INFO' in log:
                        log_class = 'text-info'
                        
                    formatted_logs.append(f'<div class="{log_class}">{log}</div>')
                
                return ''.join(formatted_logs)
            except Exception as e:
                return f'<div class="text-danger">Error reading logs: {str(e)}</div>'

        @self.api_server.route('/api/mimic-samples', methods=['GET'])
        def get_mimic_samples():
            """Get all mimic samples"""
            try:
                mimic_dir = os.path.join(self.data_dir, 'mimic')
                all_samples_file = os.path.join(mimic_dir, 'all_mimicked_samples.json')
                
                if not os.path.exists(all_samples_file):
                    return jsonify([])
                    
                with open(all_samples_file, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
                    # Fix: The file contains a direct list of samples, not a dict with 'samples' key
                    if isinstance(samples, list):
                        return jsonify(samples)
                    # Handle the case where it might be a dict with a 'samples' key
                    elif isinstance(samples, dict) and 'samples' in samples:
                        return jsonify(samples['samples'])
                    # Return an empty list as fallback
                    else:
                        return jsonify([])
            except Exception as e:
                self.logger.error(f"Error getting mimic samples: {e}")
                return jsonify([])
            
        
        @self.api_server.route('/api/harvester-insights', methods=['GET'])
        def get_harvester_insights():
            """Get harvester insights summary"""
            try:
                insights_file = os.path.join(self.data_dir, 'harvester', 'insights_summary.txt')
                
                if not os.path.exists(insights_file):
                    return "No insights summary available yet. Run the harvester component first.", 200
                    
                with open(insights_file, 'r', encoding='utf-8') as f:
                    insights_text = f.read()
                    
                return insights_text, 200
            except Exception as e:
                self.logger.error(f"Error getting harvester insights: {e}")
                return f"Error loading insights: {str(e)}", 500

        # Serve static files from the data directory
        @self.api_server.route('/carnis_data/<path:path>')
        def serve_static(path):
            """Serve static files from the data directory"""
            # First try to serve from data_dir
            if os.path.exists(os.path.join(self.data_dir, path)):
                return send_from_directory(self.data_dir, path)
            
            # Create directories for static files if they don't exist
            static_dirs = ['meatsnake', 'mimic', 'harvester']
            for dir_name in static_dirs:
                full_dir = os.path.join(self.data_dir, dir_name)
                os.makedirs(full_dir, exist_ok=True)
                
            return send_from_directory(self.data_dir, path)

        @self.api_server.route('/carnis_data/<path:path>')
        def serve_data_files(path):
            """Serve files directly from the data directory structure"""
            return send_from_directory(self.data_dir, path)
        
        # Configuration for the API server
        self.api_host = self.config.get('api_host', '127.0.0.1')
        self.api_port = self.config.get('api_port', 5000)

        @self.api_server.route('/api/component-config/<component>', methods=['POST'])
        def update_component_config_endpoint(component):
            """API endpoint to update component configuration"""
            if component not in self.components:
                return jsonify({'success': False, 'message': f'Component {component} not found'}), 404
                
            try:
                config_data = request.json
                success, message = self.update_component_config(component, config_data)
                return jsonify({'success': success, 'message': message})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 400
    
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
        # Return paths using the carnis_data directory structure
        return {
            'crawl': os.path.join('carnis_data', 'crawl', 'crawled_data.json'),
            'trimmings': os.path.join('carnis_data', 'trimmings', 'trimmed_data.json'),
            'meatsnake': os.path.join('carnis_data', 'meatsnake', 'knowledge_graph.json'),
            'mimic': os.path.join('carnis_data', 'mimic'),
            'harvester': os.path.join('carnis_data', 'harvester')
        }
    
    def update_component_config(self, component_name, config_data):
        """Update configuration for a specific component"""
        if component_name not in self.components:
            return False, f"Component {component_name} not found"
        
        try:
            # Update component config
            self.components[component_name]['config'].update(config_data)
            
            # Update the main config
            self.config['components'][component_name] = self.components[component_name]['config']
            
            # Save to file
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            # If the component is already initialized, re-initialize it
            if self.components[component_name]['initialized']:
                self.components[component_name]['initialized'] = False
                self.initialize_component(component_name)
                
            return True, "Configuration updated successfully"
        except Exception as e:
            self.logger.error(f"Error updating {component_name} config: {e}")
            return False, f"Error: {str(e)}"
    
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
                # Crawl is initialized with seed URLs
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
                # Trimmings processes the crawl output
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
    parser = argparse.ArgumentParser(description='Carnis Host System')
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