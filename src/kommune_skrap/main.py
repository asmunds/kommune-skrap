import requests
from bs4 import BeautifulSoup

__author__ = "Åsmund Frantzen Skomedal"
__copyright__ = "Åsmund Frantzen Skomedal"
__license__ = "MIT"

URL = r"https://politiskagenda.kristiansand.kommune.no/?request.kriterie.udvalgId=355a0ec6-ac1f-4e76-8ad4-3ab898841838&request.kriterie.moedeDato=2024"

def main(url: str) -> None:
    """Use beatifulsoup to scrape kommune data from the given url."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        search_button = soup.find('button', string='searchButton')
        if search_button:
            print("Found the 'searchbutton' button.")
        else:
            print("No 'searchbutton' button found.")
        resultater_div = soup.find(id='resultater')
        if resultater_div:
            links = resultater_div.find_all('a')
            for link in links:
            print(link.get('href'))
        else:
            print("No element with id 'resultater' found.")
    else:
        print(f"Failed to retrieve the URL: {url}, status code: {response.status_code}")



if __name__ == "__main__":
    main(url=URL)
