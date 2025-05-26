import json
import requests
from datetime import datetime
import os
from pathlib import Path

class ThreatIntelligenceUpdater:
    def __init__(self, config_path=None):
        self.config_path = config_path or Path(__file__).parent / 'threat_intel_config.json'
        self.db_path = Path(__file__).parent / 'threat_intelligence_db.json'
        self.demo_path = Path(__file__).parent / 'threat_intel_demo.json'
        self.load_config()
        
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'update_interval': 86400,  # 24 hours in seconds
                'api_endpoints': [],
                'last_update': None
            }
            self.save_config()
            
    def save_config(self):
        """Save configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def load_threat_db(self):
        """Load threat intelligence database"""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'mirai_variants': {},
                'attack_patterns': {},
                'known_vulnerabilities': {},
                'last_update': None
            }
            
    def save_threat_db(self, db):
        """Save threat intelligence database"""
        with open(self.db_path, 'w') as f:
            json.dump(db, f, indent=4)
            
    def update_from_api(self, endpoint, api_key=None):
        """Update threat intelligence from API endpoint"""
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Error updating from {endpoint}: {str(e)}")
            return None
            
    def merge_threat_data(self, existing_data, new_data):
        """Merge new threat data with existing data"""
        if not new_data:
            return existing_data
            
        # Merge Mirai variants
        if 'mirai_variants' in new_data:
            existing_data['mirai_variants'].update(new_data['mirai_variants'])
            
        # Merge attack patterns
        if 'attack_patterns' in new_data:
            existing_data['attack_patterns'].update(new_data['attack_patterns'])
            
        # Merge vulnerabilities
        if 'known_vulnerabilities' in new_data:
            existing_data['known_vulnerabilities'].update(new_data['known_vulnerabilities'])
            
        return existing_data
        
    def update_threat_intelligence(self):
        """Update threat intelligence from all configured sources"""
        db = self.load_threat_db()
        
        # Update from each configured API endpoint
        for endpoint in self.config.get('api_endpoints', []):
            new_data = self.update_from_api(
                endpoint['url'],
                endpoint.get('api_key')
            )
            db = self.merge_threat_data(db, new_data)
            
        # Update timestamp
        db['last_update'] = datetime.now().isoformat()
        self.save_threat_db(db)
        
        # Update config with last update time
        self.config['last_update'] = db['last_update']
        self.save_config()
        
        return db
        
    def load_demo_data(self):
        """Load demo threat intelligence data"""
        try:
            with open(self.demo_path, 'r') as f:
                demo_data = json.load(f)
            return demo_data
        except FileNotFoundError:
            print(f"Demo data file not found at {self.demo_path}")
            return None
            
    def initialize_demo_environment(self):
        """Initialize demo environment with sample threat intelligence data"""
        demo_data = {
            'mirai_variants': {
                'mirai_1': {
                    'name': 'Mirai.Generic',
                    'signature': 'tcp_syn_flood_pattern_1',
                    'severity': 'high'
                },
                'mirai_2': {
                    'name': 'Mirai.Botnet',
                    'signature': 'udp_flood_pattern_1',
                    'severity': 'critical'
                }
            },
            'attack_patterns': {
                'pattern_1': {
                    'name': 'TCP SYN Flood',
                    'characteristics': ['high_syn_rate', 'low_ack_rate'],
                    'mitigation': 'rate_limiting'
                },
                'pattern_2': {
                    'name': 'UDP Flood',
                    'characteristics': ['high_udp_rate', 'random_ports'],
                    'mitigation': 'traffic_filtering'
                }
            },
            'known_vulnerabilities': {
                'vuln_1': {
                    'cve': 'CVE-2023-1234',
                    'description': 'Buffer overflow in IoT firmware',
                    'affected_devices': ['camera_1', 'router_2']
                }
            },
            'last_update': datetime.now().isoformat()
        }
        
        # Save demo data
        with open(self.demo_path, 'w') as f:
            json.dump(demo_data, f, indent=4)
            
        return demo_data

if __name__ == "__main__":
    updater = ThreatIntelligenceUpdater()
    
    # Initialize demo environment if needed
    if not os.path.exists(updater.demo_path):
        print("Initializing demo environment...")
        updater.initialize_demo_environment()
        
    # Update threat intelligence
    print("Updating threat intelligence...")
    updated_db = updater.update_threat_intelligence()
    print(f"Threat intelligence updated. Last update: {updated_db['last_update']}") 