import aiohttp
import asyncio
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

# File to store PDF names
OUTPUT_FILE = "./pdf_names.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Global list to collect PDF names
pdf_names = []

async def process_directory(session, url, dir_sem):
    async with dir_sem:
        try:
            async with session.get(url, timeout=30) as response:
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                return [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                        if a['href'] not in ['../', './', "/pub/pmc/"]]
        except Exception as e:
            print(f"\nError processing {url}: {e}")
            return []

async def process_second_level(session, dir2, dir_sem, sub1, sub2, subdir_pbar):
    subdir_pbar.set_postfix({'Subdir': sub2}, refresh=True)
    
    # Get PDF URLs from this directory
    urls3 = await process_directory(session, dir2, dir_sem)
    pdf_urls = [url for url in urls3 if url.endswith('.pdf')]
    
    # Collect PDF names with prefixes
    for pdf in pdf_urls:
        original_fname = pdf.split('/')[-1]
        prefixed_fname = f"{sub1}___{sub2}___{original_fname}"
        pdf_names.append(prefixed_fname)
    
    subdir_pbar.update(1)

async def main():
    # Separate controls for directory and download concurrency
    conn = aiohttp.TCPConnector(limit=8000)         # Total concurrent connections
    dir_sem = asyncio.Semaphore(200)               # Concurrent directory processing
    timeout = aiohttp.ClientTimeout(total=300)
    
    print(f"Settings:")
    print(f"Total connections: 200")
    print(f"Directory concurrency: 200")
    
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        # Get root directory listing
        async with session.get("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/") as response:
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            urls1 = [urljoin("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/", a['href'])
                    for a in soup.find_all('a', href=True)
                    if a['href'] not in ['../', './', "/pub/pmc/"]]

        # Process main directories with progress bar
        main_pbar = tqdm(urls1, desc="Processing main directories")
        for dir1 in main_pbar:
            sub1 = dir1.rstrip('/').split('/')[-1]
            main_pbar.set_postfix({'Directory': sub1}, refresh=True)
            
            # Get second level directories
            urls2 = await process_directory(session, dir1, dir_sem)
            
            # Process all second level directories in parallel
            subdir_pbar = tqdm(total=len(urls2), desc=f"Processing {sub1} subdirs", leave=False)
            
            # Create tasks for all subdirectories at once
            subdir_tasks = []
            for dir2 in urls2:
                sub2 = dir2.rstrip('/').split('/')[-1]
                task = process_second_level(session, dir2, dir_sem, sub1, sub2, subdir_pbar)
                subdir_tasks.append(task)
            
            # Run all subdirectory tasks in parallel
            await asyncio.gather(*subdir_tasks)
            subdir_pbar.close()
            
        main_pbar.close()

if __name__ == "__main__":
    asyncio.run(main())
    
    # Write collected PDF names to file
    with open(OUTPUT_FILE, 'w') as f:
        for name in pdf_names:
            f.write(f"{name}\n")
    
    print(f"\n\nCompleted! Total PDF names collected: {len(pdf_names)}")
    print(f"Names saved to: {OUTPUT_FILE}")