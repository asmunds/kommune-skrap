"""
This module provides functionality to scrape and download PDF files from a specified URL.
It uses Selenium to interact with the web page and Requests to download the files.
Author:
    Åsmund Frantzen Skomedal
Copyright:
    Åsmund Frantzen Skomedal
License:
    MIT
Constants:
    URL (str): The URL to scrape data from.
    DOWNLOAD_DIR (Path): The directory to save downloaded PDF files.
    KEYWORDS (list): List of keywords to search for in the scraped data.
Functions:
    download_pdf(*, url, download_dir: Path, new_filename: str):
        Download a PDF file from the given URL to the specified directory.
    identify_keyword(*, section_title, keywords):
        Identify which keyword is present in the section title.
    main(url: str) -> None:
        Use Selenium to scrape kommune data from the given URL.
"""

import os
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

URL = r"https://politiskagenda.kristiansand.kommune.no/"
DOWNLOAD_DIR = Path("./data/")  # Directory to save downloaded PDF files
KEYWORDS = ["dispensasjon"]  # Keywords to search for in the scraped data


def download_pdf(*, url, download_dir: Path, new_filename: str):
    """Download a PDF file from the given URL to the specified directory.

    Args:
        url (str): URL to the PDF file.
        download_dir (str): Directory to save the PDF file.
    """
    response = requests.get(url)
    if response.status_code == 200:
        filepath = download_dir / (new_filename + ".pdf")
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Downloaded: {new_filename}.pdf")
    else:
        print(f"Failed to download: {new_filename}.pdf")


def identify_keyword(*, section_title, keywords):
    """Identify which keyword is present in the section title.

    Args:
        section_title (str): The title of the section.
        keywords (list): List of keywords to search for.

    Returns:
        str or None: The keyword found in the section title, or None if no keyword is found.
    """
    for keyword in keywords:
        if keyword in section_title.lower():
            return keyword
    return None


def main(url: str) -> None:
    """Use beatifulsoup to scrape kommune data from the given url.

    Args:
        url (str): URL to scrape data from.
    """
    # Create download directory if it doesn't exist (check if parent directory exists)
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR, parents=False)

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the URL: {url}, status code: {response.status_code}")
        return

    # Initialize Selenium WebDriver (make sure to have the appropriate driver installed)
    driver = webdriver.Firefox()
    driver.get(URL)

    # Wait until the dropdown is present
    wait = WebDriverWait(driver, 10)

    # Prompt the user for input to continue or exit
    while (
        input('\nMake relevant search, then press enter. Write "exit" when done. ')
        .strip()
        .lower()
        != "exit"
    ):
        # Get the results
        resultater = driver.find_element(By.ID, "resultater")

        # Find all links in the resultater table
        links = resultater.find_elements(By.TAG_NAME, "a")
        link_data_list = [
            {
                "url": link.get_attribute("href"),
                "date": link.find_element(By.CLASS_NAME, "col-sm-3").text,
            }
            for link in links
        ]

        # Loop through all links to find the relevant data
        for link_data in link_data_list:
            link_url = link_data.get("url")
            date_text = link_data.get("date").replace("/", ".")

            # Click on the link to enter a new web page
            driver.execute_script("window.open();")
            driver.switch_to.window(driver.window_handles[-1])
            driver.get(link_url)

            # Wait until the new page is loaded
            wait.until(lambda driver: driver.current_url != URL)
            wait.until(lambda driver: driver.find_element(By.TAG_NAME, "body"))
            # Wait until the number of links is greater than 2
            wait.until(lambda driver: len(driver.find_elements(By.TAG_NAME, "a")) > 2)

            # Find and click all '+' signs to reveal hidden data
            plus_signs = driver.find_elements(
                By.CSS_SELECTOR, "span.glyphicon.glyphicon-plus[aria-hidden='true']"
            )
            for plus_sign in plus_signs:
                plus_sign.click()

            # Find table called dagsordenDetaljer
            dagsorden_detaljer = driver.find_element(By.ID, "dagsordenDetaljer")

            # Get all the rows in tbody of dagsordenDetaljer
            rows = dagsorden_detaljer.find_elements(By.XPATH, ".//tbody/tr")

            # Loop through all the rows in the table to find the relevant data
            for row in rows:
                # Get the header of the current row
                section_title = row.find_element(
                    By.XPATH, ".//h2[contains(@class, 'overskrift')]"
                ).text

                # Use the identify_keyword function to check for keywords in the section title
                keyword_found = identify_keyword(
                    section_title=section_title, keywords=KEYWORDS
                )
                if keyword_found:
                    # Get the new links that have appeared after clicking the plus sign
                    # Find links that are located in this row of the table
                    subpage_links = row.find_elements(By.XPATH, ".//a")
                    subpage_links_dict = [
                        {
                            "href": link.get_attribute("href"),
                            "text": link.text,
                        }
                        for link in subpage_links
                    ]
                    # Loop through the new links to find the one that contains the PDF
                    for subpage_link in subpage_links_dict:
                        new_href = subpage_link.get("href")
                        if (
                            new_href
                            and new_href.endswith("Pdf=false")
                            and "Link til sak" in subpage_link.get("text")
                        ):
                            # Modify the URL to replace "Pdf=false" with "Pdf=true"
                            pdf_url = new_href.replace("Pdf=false", "Pdf=true")

                            # Open the pdf link in a new tab
                            driver.execute_script("window.open();")
                            driver.switch_to.window(driver.window_handles[-1])
                            driver.get(pdf_url)

                            # Wait until the PDF is loaded
                            wait.until(lambda driver: driver.current_url != link_url)

                            # Make download directory if it doesn't exist
                            download_dir = DOWNLOAD_DIR / date_text
                            if not download_dir.exists():
                                download_dir.mkdir(parents=False)

                            # Download the PDF file
                            download_pdf(
                                url=pdf_url,
                                download_dir=download_dir,
                                new_filename=section_title.replace("/", "_"),
                            )

                            # Close the PDF window/tab and switch back to the original window/tab
                            driver.close()
                            driver.switch_to.window(driver.window_handles[-1])

            driver.close()  # Close this window
            driver.switch_to.window(driver.window_handles[0])


if __name__ == "__main__":
    main(url=URL)
