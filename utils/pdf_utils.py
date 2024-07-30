from PyPDF2 import PdfReader, PdfWriter
import os, shutil


def split_pdf(file_path: str, output_dir: str, chunk_size: int = 15) -> None:
    """
    Split a PDF file into chunks and save them in the output directory.
    
    The function uses the PyPDF2 library to read the input PDF and split it into chunks.
    The function creates the output directory if it doesn't exist. 
    It then opens the PDF file using PdfFileReader and gets the total number of pages.
    It iterates over the pages in chunks, creating a new PdfFileWriter for each chunk. 
    It adds the pages to the writer and saves the chunk to a new PDF file in the output directory.
    After splitting the PDF, the function prints a message indicating the number of chunks created.
    You can customize the chunk_size parameter according to your needs.


    Args:
        file_path (str): The file path of the input PDF.
        output_dir (str): The output directory where the chunks will be saved.
        chunk_size (int, optional): The number of pages per chunk. Defaults to 15.

    Returns:
        None

    Example: split a single pdf into chunks and save to a directory
        file_path = 'path/to/input.pdf'
        output_dir = 'path/to/output_directory'
        split_pdf(file_path, output_dir)
    
    Example 2: loop over directory of pdf files, chunk them, and save the chunks in the same directory under a new folder

        # Define the directory containing the PDF files
        pdf_dir = "/home/zjc1002/Mounts/code/temp/publicCAs"
        files = os.listdir(pdf_dir)

        # Loop over each file in the directory
        for file_name in files:
            if file_name.endswith(".pdf"):
                file_path = os.path.join(pdf_dir, file_name)

                # Define the output directory for the chunks
                output_dir = os.path.join(pdf_dir, file_name.split(".")[0])

                # Split the PDF file into chunks
                split_pdf(file_path, output_dir, chunk_size=10)

    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)

        # Get the total number of pages in the PDF
        total_pages =  len(pdf.pages)

        # Split the PDF into chunks
        for start in range(0, total_pages, chunk_size):
            end = start + chunk_size if start + chunk_size <= total_pages else total_pages

            # Create a new PDF writer for each chunk
            writer = PdfWriter()

            # Add pages to the writer
            for page_num in range(start, end):
                page = pdf.pages[page_num]
                writer.add_page(page)

            # Save the chunk to a new PDF file
            output_file = os.path.join(output_dir, f'chunk_{start + 1}-{end}.pdf')
            with open(output_file, 'wb') as output:
                writer.write(output)

    print(f'PDF file "{file_path}" has been split into {total_pages // chunk_size} chunks.')
