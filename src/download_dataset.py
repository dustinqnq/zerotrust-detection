import os
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

class DatasetDownloader:
    def __init__(self):
        self.datasets = {
            'iot-23': {
                'base_url': 'https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/',
                'files': [
                    'CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled',
                    'CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled',
                    'CTU-IoT-Malware-Capture-7-1/bro/conn.log.labeled'
                ]
            }
        }
        
    def download_file(self, url, dest_path, chunk_size=8192):
        """Download a file with resume support and progress bar"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Get file size
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))
        
        # Check if file exists and get current size
        initial_pos = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        
        # Set up progress bar
        progress = tqdm(
            total=total_size,
            initial=initial_pos,
            unit='iB',
            unit_scale=True,
            desc=os.path.basename(dest_path)
        )
        
        # Set up headers for resume
        headers = {'Range': f'bytes={initial_pos}-'} if initial_pos > 0 else {}
        
        try:
            # Stream download
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                mode = 'ab' if initial_pos > 0 else 'wb'
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        size = f.write(chunk)
                        progress.update(size)
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            if os.path.exists(dest_path):
                print(f"Partial file saved at {dest_path}")
            return False
        finally:
            progress.close()
            
        return True
        
    def download_dataset(self, dataset_name, output_dir):
        """Download a specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
            
        dataset_info = self.datasets[dataset_name]
        base_url = dataset_info['base_url']
        
        for file_path in dataset_info['files']:
            url = base_url + file_path
            dest_path = os.path.join(output_dir, dataset_name, file_path)
            
            print(f"\nDownloading {file_path}...")
            success = self.download_file(url, dest_path)
            
            if success:
                print(f"Successfully downloaded to {dest_path}")
            else:
                print(f"Failed to download {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Download datasets with resume support')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['iot-23'],
                       help='Dataset to download')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    downloader.download_dataset(args.dataset, args.output_dir)

if __name__ == '__main__':
    main() 