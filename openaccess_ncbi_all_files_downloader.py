# # # import requests, os
# # # from bs4 import BeautifulSoup
# # # from urllib.parse import urljoin
# # # from tqdm import tqdm
# # # from pathlib import Path

# # # save_dir = "./data/openacess_ncbi/"
# # # os.makedirs(save_dir, exist_ok=True)
# # # session = requests.Session()
# # # pdfs = 0

# # # existing_files = {f.name for f in Path(save_dir).glob('*/*/*.pdf')}

# # # # Process each level of directories and download PDFs
# # # soup = BeautifulSoup(session.get("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/").text, 'html.parser')
# # # urls1 = [urljoin("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/", a['href']) for a in soup.find_all('a', href=True) if a['href'] not in ['../', './', "/pub/pmc/"]]
# # # for dir1 in tqdm(urls1, total=len(urls1), desc="Downloaded Root Subfolders"):
# # #     print(f"\nProcessing: {dir1}")
# # #     urls2 = [urljoin(dir1, a['href']) for a in BeautifulSoup(session.get(dir1).text, 'html.parser').find_all('a', href=True) if a['href'] not in ['../', './', "/pub/pmc/"]]
# # #     for dir2 in urls2:
# # #         try:
# # #             urls3 = [urljoin(dir2, a['href']) for a in BeautifulSoup(session.get(dir2).text, 'html.parser').find_all('a', href=True) if a['href'].endswith('.pdf') and a['href'].split('/')[-1] not in existing_files]
# # #             for pdf in urls3:
# # #                 fname = pdf.split('/')[-1]
# # #                 if not os.path.exists(os.path.join(save_dir, fname)):
# # #                     with open(os.path.join(save_dir, fname), 'wb') as f:
# # #                         f.write(session.get(pdf).content)
# # #                     pdfs += 1
# # #                     print(f"Downloaded {pdfs} PDFs | Current: {fname}", end='\r')
# # #         except Exception as e:
# # #             print(f"\nError: {e}")

# # # print(f"\n\nCompleted! PDFs downloaded: {pdfs}")


# # import aiohttp
# # import asyncio
# # import os
# # from bs4 import BeautifulSoup
# # from urllib.parse import urljoin
# # from tqdm import tqdm
# # from pathlib import Path
# # import aiofiles

# # save_dir = "./data/openacess_ncbi/"
# # os.makedirs(save_dir, exist_ok=True)
# # existing_files = {f.name for f in Path(save_dir).glob('*/*/*.pdf')}
# # pdfs_downloaded = 0

# # async def download_pdf(session, url, semaphore):
# #     global pdfs_downloaded
# #     fname = url.split('/')[-1]
# #     if fname in existing_files:
# #         return

# #     save_path = os.path.join(save_dir, fname)
# #     if os.path.exists(save_path):
# #         return

# #     async with semaphore:  # Limit concurrent downloads
# #         try:
# #             async with session.get(url) as response:
# #                 if response.status == 200:
# #                     async with aiofiles.open(save_path, 'wb') as f:
# #                         await f.write(await response.read())
# #                     pdfs_downloaded += 1
# #                     print(f"Downloaded {pdfs_downloaded} PDFs | Current: {fname}", end='\r')
# #         except Exception as e:
# #             print(f"\nError downloading {url}: {e}")

# # async def process_directory(session, url, semaphore):
# #     async with semaphore:  # Limit concurrent requests
# #         try:
# #             async with session.get(url) as response:
# #                 text = await response.text()
# #                 soup = BeautifulSoup(text, 'html.parser')
# #                 return [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
# #                         if a['href'] not in ['../', './', "/pub/pmc/"]]
# #         except Exception as e:
# #             print(f"\nError processing {url}: {e}")
# #             return []

# # async def main():
# #     # Configure connection pooling and concurrent limits
# #     conn = aiohttp.TCPConnector(limit=100)  # Limit concurrent connections
# #     timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
# #     semaphore = asyncio.Semaphore(100)  # Limit concurrent operations
    
# #     async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
# #         # Get root directory listing
# #         async with session.get("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/") as response:
# #             text = await response.text()
# #             soup = BeautifulSoup(text, 'html.parser')
# #             urls1 = [urljoin("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/", a['href'])
# #                     for a in soup.find_all('a', href=True)
# #                     if a['href'] not in ['../', './', "/pub/pmc/"]]

# #         # Process main directories
# #         for dir1 in tqdm(urls1, desc="Processing main directories"):
# #             print(f"\nProcessing: {dir1}")
            
# #             # Get second level directories
# #             urls2 = await process_directory(session, dir1, semaphore)
            
# #             # Process subdirectories concurrently
# #             subdir_tasks = [process_directory(session, dir2, semaphore) for dir2 in urls2]
# #             subdir_results = await asyncio.gather(*subdir_tasks)
            
# #             # Collect PDF URLs
# #             pdf_urls = []
# #             for urls3 in subdir_results:
# #                 pdf_urls.extend([url for url in urls3 if url.endswith('.pdf') 
# #                                and url.split('/')[-1] not in existing_files])
            
# #             # Download PDFs concurrently
# #             download_tasks = [download_pdf(session, pdf, semaphore) for pdf in pdf_urls]
# #             await asyncio.gather(*download_tasks)

# # if __name__ == "__main__":
# #     asyncio.run(main())
# #     print(f"\n\nCompleted! PDFs downloaded: {pdfs_downloaded}")

# import aiohttp
# import asyncio
# import os
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from tqdm import tqdm
# from pathlib import Path
# import aiofiles

# save_dir = "./data/openacess_ncbi_v2/"
# os.makedirs(save_dir, exist_ok=True)
# pdfs_downloaded = 0

# async def download_pdf(session, url, semaphore, sub1, sub2):
#     global pdfs_downloaded
#     original_fname = url.split('/')[-1]
#     prefixed_fname = f"{sub1}___{sub2}___{original_fname}"
    
#     save_path = os.path.join(save_dir, prefixed_fname)
#     if os.path.exists(save_path):
#         return

#     async with semaphore:
#         try:
#             async with session.get(url) as response:
#                 if response.status == 200:
#                     async with aiofiles.open(save_path, 'wb') as f:
#                         await f.write(await response.read())
#                     pdfs_downloaded += 1
#                     print(f"Downloaded {pdfs_downloaded} PDFs | Current: {prefixed_fname}", end='\r')
#         except Exception as e:
#             print(f"\nError downloading {url}: {e}")

# async def process_directory(session, url, semaphore):
#     async with semaphore:
#         try:
#             async with session.get(url) as response:
#                 text = await response.text()
#                 soup = BeautifulSoup(text, 'html.parser')
#                 return [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
#                         if a['href'] not in ['../', './', "/pub/pmc/"]]
#         except Exception as e:
#             print(f"\nError processing {url}: {e}")
#             return []

# async def main():
#     conn = aiohttp.TCPConnector(limit=10)
#     timeout = aiohttp.ClientTimeout(total=300)
#     semaphore = asyncio.Semaphore(1)
    
#     async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
#         # Get root directory listing
#         async with session.get("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/") as response:
#             text = await response.text()
#             soup = BeautifulSoup(text, 'html.parser')
#             urls1 = [urljoin("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/", a['href'])
#                     for a in soup.find_all('a', href=True)
#                     if a['href'] not in ['../', './', "/pub/pmc/"]]

#         # Process main directories
#         for dir1 in tqdm(urls1, desc="Processing main directories"):
#             sub1 = dir1.rstrip('/').split('/')[-1]  # Get first subdirectory name
#             print(f"\nProcessing: {dir1}")
            
#             # Get second level directories
#             urls2 = await process_directory(session, dir1, semaphore)
            
#             # Process each second level directory
#             for dir2 in urls2:
#                 sub2 = dir2.rstrip('/').split('/')[-1]  # Get second subdirectory name
                
#                 # Get PDF URLs from this directory
#                 urls3 = await process_directory(session, dir2, semaphore)
#                 pdf_urls = [url for url in urls3 if url.endswith('.pdf')]
                
#                 # Download PDFs concurrently with subdirectory names
#                 download_tasks = [download_pdf(session, pdf, semaphore, sub1, sub2) 
#                                 for pdf in pdf_urls]
#                 await asyncio.gather(*download_tasks)

# if __name__ == "__main__":
#     asyncio.run(main())
#     print(f"\n\nCompleted! PDFs downloaded: {pdfs_downloaded}")


# import aiohttp
# import asyncio
# import os
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from tqdm import tqdm
# from pathlib import Path
# import aiofiles

# save_dir = "./data/openacess_ncbi_v2/"
# os.makedirs(save_dir, exist_ok=True)
# pdfs_downloaded = 0

# async def download_pdf(session, url, semaphore, sub1, sub2, pbar):
#     global pdfs_downloaded
#     original_fname = url.split('/')[-1]
#     prefixed_fname = f"{sub1}___{sub2}___{original_fname}"
#     save_path = os.path.join(save_dir, prefixed_fname)
    
#     if os.path.exists(save_path):
#         pbar.update(1)
#         return
        
#     async with semaphore:
#         try:
#             async with session.get(url) as response:
#                 if response.status == 200:
#                     async with aiofiles.open(save_path, 'wb') as f:
#                         await f.write(await response.read())
#                     pdfs_downloaded += 1
#                     pbar.update(1)
#                     pbar.set_postfix({'Current': prefixed_fname[:30]}, refresh=True)
#         except Exception as e:
#             print(f"\nError downloading {url}: {e}")
#             pbar.update(1)

# async def process_directory(session, url, semaphore):
#     async with semaphore:
#         try:
#             async with session.get(url) as response:
#                 text = await response.text()
#                 soup = BeautifulSoup(text, 'html.parser')
#                 return [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
#                         if a['href'] not in ['../', './', "/pub/pmc/"]]
#         except Exception as e:
#             print(f"\nError processing {url}: {e}")
#             return []

# async def main():
#     conn = aiohttp.TCPConnector(limit=64)
#     timeout = aiohttp.ClientTimeout(total=300)
#     semaphore = asyncio.Semaphore(64)
    
#     async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
#         # Get root directory listing
#         async with session.get("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/") as response:
#             text = await response.text()
#             soup = BeautifulSoup(text, 'html.parser')
#             urls1 = [urljoin("https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/", a['href'])
#                     for a in soup.find_all('a', href=True)
#                     if a['href'] not in ['../', './', "/pub/pmc/"]]

#         # Process main directories with progress bar
#         main_pbar = tqdm(urls1, desc="Processing main directories")
#         for dir1 in main_pbar:
#             sub1 = dir1.rstrip('/').split('/')[-1]  # Get first subdirectory name
#             main_pbar.set_postfix({'Directory': sub1}, refresh=True)
            
#             # Get second level directories
#             urls2 = await process_directory(session, dir1, semaphore)
            
#             # Process each second level directory with progress bar
#             subdir_pbar = tqdm(urls2, desc=f"Processing {sub1} subdirs", leave=False)
#             for dir2 in subdir_pbar:
#                 sub2 = dir2.rstrip('/').split('/')[-1]  # Get second subdirectory name
#                 subdir_pbar.set_postfix({'Subdir': sub2}, refresh=True)
                
#                 # Get PDF URLs from this directory
#                 urls3 = await process_directory(session, dir2, semaphore)
#                 pdf_urls = [url for url in urls3 if url.endswith('.pdf')]
                
#                 # Create progress bar for PDFs in this directory
#                 pdf_pbar = tqdm(total=len(pdf_urls), desc=f"Downloading PDFs from {sub2}", leave=False)
                
#                 # Download PDFs concurrently with subdirectory names
#                 download_tasks = [download_pdf(session, pdf, semaphore, sub1, sub2, pdf_pbar)
#                                 for pdf in pdf_urls]
#                 await asyncio.gather(*download_tasks)
#                 pdf_pbar.close()
#             subdir_pbar.close()
#         main_pbar.close()

# if __name__ == "__main__":
#     asyncio.run(main())
#     print(f"\n\nCompleted! PDFs downloaded: {pdfs_downloaded}")


import aiohttp
import asyncio
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from pathlib import Path
import aiofiles

save_dir = "./data/openacess_ncbi_v5/"
os.makedirs(save_dir, exist_ok=True)
pdfs_downloaded = 0

async def download_pdf(session, url, download_sem, sub1, sub2, pbar):
    global pdfs_downloaded
    original_fname = url.split('/')[-1]
    prefixed_fname = f"{sub1}___{sub2}___{original_fname}"
    save_path = os.path.join(save_dir, prefixed_fname)
    
    if os.path.exists(save_path):
        pbar.update(1)
        return
        
    async with download_sem:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    async with aiofiles.open(save_path, 'wb') as f:
                        await f.write(await response.read())
                    pdfs_downloaded += 1
                    pbar.update(1)
                    pbar.set_postfix({'Current': prefixed_fname[:30]}, refresh=True)
        except Exception as e:
            print(f"\nError downloading {url}: {e}")
            pbar.update(1)

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

async def process_second_level(session, dir2, dir_sem, download_sem, sub1, sub2, subdir_pbar):
    subdir_pbar.set_postfix({'Subdir': sub2}, refresh=True)
    
    # Get PDF URLs from this directory
    urls3 = await process_directory(session, dir2, dir_sem)
    pdf_urls = [url for url in urls3 if url.endswith('.pdf')]
    
    # Create progress bar for PDFs in this directory
    pdf_pbar = tqdm(total=len(pdf_urls), desc=f"Downloading PDFs from {sub2}", leave=False)
    
    # Download PDFs concurrently
    download_tasks = [download_pdf(session, pdf, download_sem, sub1, sub2, pdf_pbar)
                     for pdf in pdf_urls]
    await asyncio.gather(*download_tasks)
    pdf_pbar.close()
    subdir_pbar.update(1)

async def main():
    # Separate controls for directory and download concurrency
    conn = aiohttp.TCPConnector(limit=8000)         # Total concurrent connections
    dir_sem = asyncio.Semaphore(200)               # Concurrent directory processing
    download_sem = asyncio.Semaphore(40)          # Concurrent PDF downloads
    timeout = aiohttp.ClientTimeout(total=300)
    
    print(f"Settings:")
    print(f"Total connections: 200")
    print(f"Directory concurrency: 40")
    print(f"Download concurrency: 200")
    
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
                task = process_second_level(session, dir2, dir_sem, download_sem, sub1, sub2, subdir_pbar)
                subdir_tasks.append(task)
            
            # Run all subdirectory tasks in parallel
            await asyncio.gather(*subdir_tasks)
            subdir_pbar.close()
            
        main_pbar.close()

if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n\nCompleted! PDFs downloaded: {pdfs_downloaded}")