# Carnis

A modular, hierarchical web data processing system that crawls, processes, analyzes, and generates insights from web content.

## Overview

Carnis is a sophisticated data harvesting and processing system with a biological metaphor at its core, based on Darian Quilloy's *Vita Carnis* analog horror series. The system is structured in a hierarchical manner, with components working together like organisms in a symbiotic relationship:

- **Crawl**: Gathers raw data from the web while respecting robots.txt rules
- **Trimmings**: Processes and cleans the raw data
- **Meatsnake**: Builds knowledge graphs from processed data
- **Mimic**: Generates content based on the knowledge graphs
- **Harvester**: Extracts insights from the mimicked content and by extension the original content
- **Host**: Integrates and controls all components into one interface
- **Monolith**: Provides meta-control and multi-host integration

## Architecture

```
Monolith (Higher abstraction)
   │
   ├── Host (Control system)
   │    │
   │    ├── Crawl (Data gathering)
   │    ├── Trimmings (Data processing)
   │    ├── Meatsnake (Knowledge graph building)
   │    ├── Mimic (Content generation)
   │    └── Harvester (Insight extraction)
   │
   └── Singularity (Next evolution - conceptual)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/L4w1i3t/Carnis.git
cd Carnis
```

2. Install required dependencies:
```bash
pip install requests beautifulsoup4 flask pyyaml
```

## Configuration

The system uses YAML configuration files:

- `config.yaml`: Controls Host and component behavior
- `monolith_config.yaml`: Controls Monolith behavior (if using multiple hosts)

Create the default configuration by running the Host or Monolith component for the first time.

## Usage

### Basic Usage

Run the Host system with default settings:

```bash
python host.py
```

Run a full processing cycle:

```bash
python host.py --run-cycle
```

Run a specific component:

```bash
python host.py --run-component crawl
```

### Advanced Usage

Run the Monolith system (for multi-host setups):

```bash
python monolith.py
```

Integrate knowledge across hosts:

```bash
python monolith.py --integrate
```

### Web Interface

When running the Host with API enabled (default), access the web interface at:

```
http://127.0.0.1:5000/
```

This provides status monitoring and control options for the system.

## Components

### Crawler (`crawl.py`)
- Web crawler that fetches content from seed URLs
- Respects robots.txt rules
- Configurable depth and delay settings

### Trimmings (`trimmings.py`)
- Processes raw crawled data
- Cleans and normalizes text
- Extracts key information

### Meatsnake (`meatsnake.py`) 
- Builds knowledge graphs from processed data
- Identifies relationships between concepts
- Structures information for later use

### Mimic (`mimic.py`)
- Generates new content based on the knowledge graph
- Produces content that mimics patterns in the source data
- Creates variations with controlled randomness

### Harvester (`harvester.py`)
- Extracts insights from mimicked content
- Identifies patterns and trends
- Produces structured outputs for analysis

### Host (`host.py`)
- Integrates all components into a unified system
- Provides API and web interface
- Manages processing cycles

### Monolith (`monolith.py`)
- Higher-level abstraction
- Can manage multiple Hosts
- Integrates knowledge across systems
- Identifies meta-patterns

## Development

Carnis follows a modular design pattern where each component can work independently but gains power when integrated.

## License

This project is licensed under the terms of the included LICENSE file.

## Credits

- Created by [L4w1i3t](https://github.com/L4w1i3t)
- Source material by Darian Quilloy(https://www.youtube.com/playlist?list=PLoQCowtS-bYLdCasDSl0rMqEfcswN2L3Q)

## Last Updated
2025-03-04