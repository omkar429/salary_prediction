import pandas as pd
import requests
from bs4 import BeautifulSoup
import pathlib
import sys
import yaml

def web_scrape(page_number) -> pd.DataFrame:

    company_names = []
    allas = []
    company_reviews = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    for page in range(page_number):

        url = f'https://www.ambitionbox.com/list-of-companies?campaign=desktop_nav&page={page}'
        print("Scraping:", url)

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'lxml')

        cards = soup.find_all('div', class_='companyCardWrapper')

        for card in cards:
            name = card.find('h2', class_='companyCardWrapper__companyName')
            info = card.find('div', class_='companyCardWrapper__tertiaryInformation')
            rating = card.find('div', class_='rating_star_container')

            company_names.append(name.text.strip() if name else None)
            allas.append(info.text.strip() if info else None)
            company_reviews.append(rating.text.strip() if rating else None)

    df = pd.DataFrame({
        'company_names': company_names,
        'allas': allas,
        'company_reviews': company_reviews
    })

    return df

def save_data(data: pd.DataFrame,path: str) -> None:
    pathlib.Path(path).mkdir(parents=True,exist_ok=True)
    data.to_csv(path / 'web_data.csv',index=False)


def main() -> None:
    home_path = pathlib.Path(__file__).parent.parent.parent
    data_path_file = sys.argv[1]
    params = yaml.safe_load(open('params.yaml'))['web_scrap']
    data_path = home_path / data_path_file
    df = web_scrape(page_number=params['page_number'])
    save_data(data=df, path=data_path)


if __name__ == '__main__':
    main()
